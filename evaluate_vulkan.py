from utils.train_util import render
# from models.vertex_color import Model
import pickle
import torch
from tqdm import tqdm
from pathlib import Path
import imageio
import numpy as np
from data import loader
from utils import test_util
from utils.args import Args
from utils import cam_util
import mediapy
from icecream import ic
from vulkan_renderer.slangpy_renderer import init_slangpy_device, render_with_slangpy
# Assuming Camera and Model from slangpy_renderer will be used or adapted
from vulkan_renderer.slangpy_renderer import Camera as SlangCamera
from vulkan_renderer.slangpy_renderer import Model as SlangModel

args = Args()
args.tile_size = 16
args.image_folder = "images_4"
args.dataset_path = Path("/optane/nerf_datasets/360/bicycle")
args.output_path = Path("output/test/")
args.eval = True
args.use_ply = False
args = Args.from_namespace(args.get_parser().parse_args())

device = torch.device('cuda')
# Initialize slangpy device
slang_device = init_slangpy_device()
if not slang_device:
    raise RuntimeError("Failed to initialize slangpy device.")

if args.use_ply:
    from models.tet_color import Model as TetColorModel
    model = TetColorModel.load_ply(args.output_path / "ckpt.ply", device)
    if not hasattr(model, 'min_t'):
        model.min_t = getattr(args, 'min_t', 0.01)
else:
    from models.ingp_color import Model as IngpColorModel
    model = IngpColorModel.load_ckpt(args.output_path, device)

# model.light_offset = -1
train_cameras_orig, test_cameras_orig, scene_info = loader.load_dataset(
    args.dataset_path, args.image_folder, data_device="cuda", eval=args.eval)

slang_model_instance = SlangModel(
    vertices=model.vertices.float(), # Ensure float
    indices=model.indices.int(),     # Ensure int
    min_t_val=float(model.min_t)
)

# The actual NeRF model (model) will be passed to extract_render_data_from_model
# in slangpy_renderer.py to get SH, density, etc.

def adapt_camera_to_slang(camera_orig, default_args_dims):
    # default_args_dims: (args.image_height, args.image_width) as a fallback

    img_h, img_w = -1, -1

    if hasattr(camera_orig, 'image_height') and camera_orig.image_height > 0:
        img_h = int(camera_orig.image_height)
    
    if hasattr(camera_orig, 'image_width') and camera_orig.image_width > 0:
        img_w = int(camera_orig.image_width)

    if img_h <= 0:
        print(f"Warning: camera_orig missing valid image_height. Attempting fallback.")
        if default_args_dims and default_args_dims[0] is not None and default_args_dims[0] > 0:
            img_h = int(default_args_dims[0])
            print(f"Using args.image_height as fallback: {img_h}")
        else:
            img_h = 256 # Last resort default
            print(f"Using hardcoded default height: {img_h}")

    if img_w <= 0:
        print(f"Warning: camera_orig missing valid image_width. Attempting fallback.")
        if default_args_dims and default_args_dims[1] is not None and default_args_dims[1] > 0:
            img_w = int(default_args_dims[1])
            print(f"Using args.image_width as fallback: {img_w}")
        else:
            img_w = 256 # Last resort default
            print(f"Using hardcoded default width: {img_w}")

    # FoV
    fovx = camera_orig.FoVx if hasattr(camera_orig, 'FoVx') else np.deg2rad(60) # Default FoV
    fovy = camera_orig.FoVy if hasattr(camera_orig, 'FoVy') else np.deg2rad(60) # Default FoV

    # Camera pose (view-to-world) and center
    # Based on common NeRF camera conventions (e.g., from LLFF/NeRF++)
    # R is often world-to-view rotation, T is camera position in world.
    # Pose matrix (view_to_world) should be [R_v2w | C_w]
    # If camera_orig.R is R_w2v and camera_orig.T is C_w:
    # R_v2w = R_w2v.T
    # So, pose_matrix[:3,:3] = R_w2v.T and pose_matrix[:3,3] = C_w

    if hasattr(camera_orig, 'world_view_transform'): # If it's already view-to-world (camera pose)
        view_to_world_transform = camera_orig.world_view_transform.float()
        # camera_center might need to be extracted from it: C = -R_v2w^T * t_v2w or directly if available
        if hasattr(camera_orig, 'camera_center'):
            camera_center = camera_orig.camera_center.float().squeeze()
        else: # Estimate from pose
            # Assuming world_view_transform is [R|t] where t is translation part of V2W
            # Camera center in world C = -R.T @ t (if R is V2W rotation, t is V2W translation)
            # Or, if it's a standard pose matrix, the last column's xyz is the camera center in world coords
            # after homogeneous transform. For [R|C] type pose, C is camera center.
            # Let's assume it's [R_v2w | C_w] style, so C_w is the translation part.
            camera_center = view_to_world_transform[:3, 3].squeeze()

    elif hasattr(camera_orig, 'R') and hasattr(camera_orig, 'T'):
        # Assume R is World-to-View rotation, T is Camera Center in World
        R_w2v = camera_orig.R.float()
        C_w = camera_orig.T.float().squeeze()
        
        pose_matrix = torch.eye(4, device=device, dtype=torch.float)
        pose_matrix[:3, :3] = R_w2v.T # R_v2w = R_w2v.T
        pose_matrix[:3, 3] = C_w
        view_to_world_transform = pose_matrix
        camera_center = C_w
    else:
        print("Warning: Camera lacks R, T, or world_view_transform. Using identity pose.")
        view_to_world_transform = torch.eye(4, device=device, dtype=torch.float)
        camera_center = torch.zeros(3, device=device, dtype=torch.float)

    return SlangCamera(
        image_height=img_h,
        image_width=img_w,
        fovx=float(fovx),
        fovy=float(fovy),
        world_view_transform=view_to_world_transform.to(device), # FloatTensor[4,4]
        camera_center=camera_center.to(device)                   # FloatTensor[3]
    )

with torch.no_grad():
    epath_orig = cam_util.generate_cam_path(train_cameras_orig, 400)
    eimages = []
    # Prepare default dimensions from args once
    default_dims_from_args = (getattr(args, 'image_height', -1), getattr(args, 'image_width', -1))

    for camera_orig in tqdm(epath_orig):
        slang_camera_instance = adapt_camera_to_slang(camera_orig, default_dims_from_args)

        render_pkg = render_with_slangpy(
            slang_camera_instance,
            model, # Pass the original NeRF model here
            slang_model_instance, # Pass the SlangModel with V/I here
            slang_device,
            bg_color_rgb=(0.0, 0.0, 0.0)
        )
        if render_pkg is None:
            print("Slangpy rendering failed for a frame.")
            # Create a dummy black image to proceed
            image = np.zeros((args.image_height, args.image_width, 3), dtype=np.float32)
        else:
            image = render_pkg['render'] # Expected CHW (3, H, W)
            # Convert CHW to HWC for mediapy and test_util
            image = image.transpose(1, 2, 0) 
        
        # image = image.permute(1, 2, 0) # Original permute, check if still needed
                                        # render_with_slangpy returns CHW, transpose(1,2,0) makes it HWC.
                                        # mediapy.write_video expects HWC.
        image = image.detach().cpu().numpy() if isinstance(image, torch.Tensor) else np.asarray(image)
        eimages.append(image)

mediapy.write_video(args.output_path / "rotating_slangpy.mp4", eimages)

if args.render_train:
    splits_orig = zip(['train', 'test'], [train_cameras_orig, test_cameras_orig])
else:
    splits_orig = zip(['test'], [test_cameras_orig])

# Adapt test_util.evaluate_and_save
# This will also need to use the slangpy renderer
# For now, this part is NOT updated and will use the old renderer or fail if 'render' func is removed.
# We'd need a new evaluate_and_save_slangpy or adapt the existing one.

print("Skipping test_util.evaluate_and_save as it's not yet adapted for slangpy_renderer.")
# test_util.evaluate_and_save(model, splits_orig, args.output_path, args.tile_size, min_t=model.min_t)

# Save to PLY might also need adjustment if SlangModel is primary
# model.save2ply(Path('test_slangpy.ply')) # If 'model' is the original NeRF model
# Or, if slang_model_instance is what needs saving, it needs a save2ply method.
print("Skipping model.save2ply as its compatibility with slangpy changes needs review.")

if slang_device:
    slang_device.close()
    print("Slangpy device closed.")

# model.save2ply(Path('test.ply')) # Original save
