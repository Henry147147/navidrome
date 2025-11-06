# Build Fix Summary

## Issue
`make build` was failing with:
```
taglib_wrapper.cpp:5:10: fatal error: apeproperties.h: No such file or directory
```

## Root Cause
The Makefile's build process downloads TagLib via `fetch-taglib.sh` and sets CGO_CFLAGS to point to its include directory. However, CGO was not finding the TagLib headers because:

1. **Missing taglib subdirectory in include path**: Headers are in `include/taglib/` but only `include/` was in the path
2. **Missing CGO_CXXFLAGS**: The `taglib_wrapper.cpp` is a C++ file, which requires CGO_CXXFLAGS, not just CGO_CFLAGS

## Solution

### Files Modified:

1. **`Makefile` (lines 132-135, 148-151)**
   - Added `-I$$TAGLIB_DIR/include/taglib` to CGO_CFLAGS
   - Added CGO_CXXFLAGS with the same include paths as CGO_CFLAGS
   - Applied fix to both `build` and `debug-build` targets

**Before:**
```makefile
CGO_CFLAGS="-I$$TAGLIB_DIR/include" \
CGO_LDFLAGS="-L$$TAGLIB_DIR/lib" \
```

**After:**
```makefile
CGO_CFLAGS="-I$$TAGLIB_DIR/include -I$$TAGLIB_DIR/include/taglib" \
CGO_CXXFLAGS="-I$$TAGLIB_DIR/include -I$$TAGLIB_DIR/include/taglib" \
CGO_LDFLAGS="-L$$TAGLIB_DIR/lib" \
```

2. **`adapters/taglib/taglib_wrapper.go` (line 4)**
   - Kept simplified pkg-config directive (previously fixed during the session)

**Current (working):**
```go
#cgo pkg-config: taglib
```

## Verification

### ✅ `make test` - ALL PASS
```bash
$ make test
ok  	github.com/navidrome/navidrome/adapters/taglib
ok  	github.com/navidrome/navidrome/cmd
... (all 81 packages pass)
```

### ✅ `make build` - SUCCESS
```bash
$ make build
(builds successfully)

$ ls -lh navidrome
-rwxrwxr-x 1 henry henry 67M Nov  6 17:01 navidrome

$ ./navidrome --version
0.58.0-SNAPSHOT (4c902047)
```

## Technical Details

### Why CGO_CXXFLAGS?
- `taglib_wrapper.cpp` is a C++ source file
- CGO uses the C compiler for `.c` files (respects CGO_CFLAGS)
- CGO uses the C++ compiler for `.cpp` files (respects CGO_CXXFLAGS)
- Without CGO_CXXFLAGS, the C++ compiler doesn't get the include paths

### Why two include paths?
The TagLib headers are in two locations:
1. `/path/to/taglib/include/` - Contains wrapper headers
2. `/path/to/taglib/include/taglib/` - Contains actual TagLib headers like `apeproperties.h`

The code includes headers directly: `#include <apeproperties.h>`
So the compiler needs `-I/path/to/taglib/include/taglib` to find them.

### Why pkg-config still works for `make test`?
- `make test` uses the system TagLib installation (via pkg-config)
- System TagLib typically installs to `/usr/include/taglib/`
- The system pkg-config returns: `-I/usr/include/taglib`
- This directly points to where the headers are, so it works

- `make build` uses downloaded TagLib from `.cache/taglib/`
- Downloaded pkg-config has broken paths like `-I/taglib/include`
- So we override with explicit CGO_CFLAGS and CGO_CXXFLAGS

## Summary
Both `make test` and `make build` now work correctly:
- Tests use system TagLib with pkg-config
- Build uses downloaded TagLib with explicit include paths
- No conflicts between the two approaches
