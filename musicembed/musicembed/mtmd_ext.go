package musicembed

import (
	"unsafe"

	"github.com/hybridgroup/yzma/pkg/mtmd"
	"github.com/jupiterrider/ffi"
)

var (
	encodeChunkFunc   ffi.Fun
	getOutputEmbdFunc ffi.Fun
	mtmdExtLib        ffi.Lib
)

// loadMTMDExtensions loads additional mtmd functions not yet wrapped by yzma.
func loadMTMDExtensions(libPath string) error {
	var err error

	mtmdExtLib, err = ffi.Load(libPath + "/libmtmd.so")
	if err != nil {
		return err
	}

	encodeChunkFunc, err = mtmdExtLib.Prep("mtmd_encode_chunk",
		&ffi.TypeSint32,
		&ffi.TypePointer,
		&ffi.TypePointer)
	if err != nil {
		return err
	}

	getOutputEmbdFunc, err = mtmdExtLib.Prep("mtmd_get_output_embd",
		&ffi.TypePointer,
		&ffi.TypePointer)
	if err != nil {
		return err
	}

	return nil
}

// encodeChunk encodes an audio/image chunk through the mmproj projector.
// Returns 0 on success.
func encodeChunk(ctx mtmd.Context, chunk mtmd.InputChunk) int32 {
	var result ffi.Arg
	encodeChunkFunc.Call(unsafe.Pointer(&result),
		unsafe.Pointer(&ctx),
		unsafe.Pointer(&chunk))
	return int32(result)
}

// getOutputEmbd returns embeddings from the last encode operation.
// Size should be: llama.ModelNEmbd(model) * mtmd.InputChunkGetNTokens(chunk)
// Note: The returned slice points to internal buffer - copy before next encode.
func getOutputEmbd(ctx mtmd.Context, size int) []float32 {
	var embdPtr *float32
	getOutputEmbdFunc.Call(unsafe.Pointer(&embdPtr),
		unsafe.Pointer(&ctx))

	if embdPtr == nil {
		return nil
	}
	return unsafe.Slice(embdPtr, size)
}
