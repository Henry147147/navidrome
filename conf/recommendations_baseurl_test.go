package conf

import "testing"

func TestAlignRecommendationBaseURLsRespectsExplicitTextBaseURL(t *testing.T) {
	original := Server
	t.Cleanup(func() { Server = original })

	Server = &configOptions{}
	Server.Recommendations.BaseURL = "http://127.0.0.1:9002"
	Server.Recommendations.TextBaseURL = "http://127.0.0.1:9003"
	Server.Recommendations.BatchBaseURL = ""

	alignRecommendationBaseURLs()

	if got := Server.Recommendations.TextBaseURL; got != "http://127.0.0.1:9003" {
		t.Fatalf("expected explicit text base URL to be preserved, got %q", got)
	}
	if got := Server.Recommendations.BatchBaseURL; got != "http://127.0.0.1:9002" {
		t.Fatalf("expected batch base URL to default to base URL, got %q", got)
	}
}

func TestAlignRecommendationBaseURLsDefaultsTextWhenUnset(t *testing.T) {
	original := Server
	t.Cleanup(func() { Server = original })

	Server = &configOptions{}
	Server.Recommendations.BaseURL = "http://127.0.0.1:9002"
	Server.Recommendations.TextBaseURL = ""
	Server.Recommendations.BatchBaseURL = ""

	alignRecommendationBaseURLs()

	if got := Server.Recommendations.TextBaseURL; got != "http://127.0.0.1:9002" {
		t.Fatalf("expected text base URL default to base URL, got %q", got)
	}
	if got := Server.Recommendations.BatchBaseURL; got != "http://127.0.0.1:9002" {
		t.Fatalf("expected batch base URL default to base URL, got %q", got)
	}
}
