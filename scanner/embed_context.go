package scanner

import "context"

type embedWaitKey struct{}

// WithEmbeddingWait marks the context so the scanner will wait for embeddings to finish.
func WithEmbeddingWait(ctx context.Context) context.Context {
	return context.WithValue(ctx, embedWaitKey{}, true)
}

func embeddingWaitEnabled(ctx context.Context) bool {
	enabled, ok := ctx.Value(embedWaitKey{}).(bool)
	return ok && enabled
}
