package subsonic

import (
	"net/http"

	"github.com/navidrome/navidrome/server/subsonic/responses"
)

/* Creates playlist from recent listens */
func (api *Router) MakePlaylistFromRecentListens(r *http.Request) (*responses.Subsonic, error) {
	//TODO
	response := newResponse()
	return response, nil
}

/* Creates playlist from other playlists */
func (api *Router) MakePlaylistFromOtherPlaylists(r *http.Request) (*responses.Subsonic, error) {
	//TODO
	response := newResponse()
	return response, nil
}

/*
Makes playlist from both the recent and the stars/hearts
*/
func (api *Router) MakePlaylistFromAllMetrics(r *http.Request) (*responses.Subsonic, error) {
	//TODO
	response := newResponse()
	return response, nil
}

/* Creates playlist from all metrics, but slightly perturbs it so its more of a discovery playlist */
func (api *Router) MakeDiscoveryPlaylist(r *http.Request) (*responses.Subsonic, error) {
	// TODO
	response := newResponse()
	return response, nil
}
