#ifndef EXPERIMENTAL_GSPLINE_VIEWER_PLY_FILE_LOADER_H_
#define EXPERIMENTAL_GSPLINE_VIEWER_PLY_FILE_LOADER_H_

#include "absl/strings/string_view.h"
#include "scene.h"

namespace gspline {

// Reads scene from a PLY file.
GaussianScene
ReadSceneFromFile(absl::string_view filename,
                  bool approximate_morton_order = true);
} // namespace gspline

#endif // EXPERIMENTAL_GSPLINE_VIEWER_PLY_FILE_LOADER_H_
