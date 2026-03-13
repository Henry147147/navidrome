package conf

import (
	"testing"

	"github.com/spf13/viper"
)

func TestMilvusDimensionsDefaultToCanonicalMuQValues(t *testing.T) {
	ResetConf()
	viper.Reset()
	SetViperDefaults()

	if got := viper.GetInt("recommendations.milvus.dimensions.muqaudio"); got != 1024 {
		t.Fatalf("expected muq_audio default 1024, got %d", got)
	}
	if got := viper.GetInt("recommendations.milvus.dimensions.muqmulan"); got != 512 {
		t.Fatalf("expected muq_mulan default 512, got %d", got)
	}
}

func TestResolvedMilvusDimensionsPreferCanonicalAndOnlyUseExplicitLegacyFallbacks(t *testing.T) {
	dims := milvusDimensions{}
	if got := dims.ResolvedMuQAudio(); got != 1024 {
		t.Fatalf("expected canonical audio fallback 1024, got %d", got)
	}
	if got := dims.ResolvedMuQMulan(); got != 512 {
		t.Fatalf("expected canonical shared fallback 512, got %d", got)
	}

	dims = milvusDimensions{Flamingo: 2048, Lyrics: 384, Description: 768}
	if got := dims.ResolvedMuQAudio(); got != 2048 {
		t.Fatalf("expected explicit flamingo fallback 2048, got %d", got)
	}
	if got := dims.ResolvedMuQMulan(); got != 768 {
		t.Fatalf("expected explicit description fallback 768, got %d", got)
	}

	dims = milvusDimensions{MuQAudio: 1536, MuQMulan: 640, Flamingo: 2048, Description: 768}
	if got := dims.ResolvedMuQAudio(); got != 1536 {
		t.Fatalf("expected canonical audio override 1536, got %d", got)
	}
	if got := dims.ResolvedMuQMulan(); got != 640 {
		t.Fatalf("expected canonical shared override 640, got %d", got)
	}
}
