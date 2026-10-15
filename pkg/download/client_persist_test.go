package download_test

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/moov-io/base/log"
	"github.com/moov-io/watchman/pkg/download"

	"github.com/stretchr/testify/require"
)

// TestClient_GetFiles_PersistsIntoMarkedCandidateDir verifies fetched bytes are
// pinned into a marked candidate directory and the returned reader sees them.
func TestClient_GetFiles_PersistsIntoMarkedCandidateDir(t *testing.T) {
	const body = "pinned-list-bytes"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(srv.Close)

	dir := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(dir, ".watchman-candidate"), []byte("gen"), 0o600))

	dl := download.New(log.NewTestLogger(), nil)
	files, err := dl.GetFiles(context.Background(), dir, map[string]string{"list.csv": srv.URL})
	require.NoError(t, err)
	t.Cleanup(func() { files.Close() })

	rc, ok := files["list.csv"]
	require.True(t, ok)
	got, err := io.ReadAll(rc)
	require.NoError(t, err)
	require.Equal(t, body, string(got))
	require.NoError(t, rc.Close())

	// Bytes are pinned on disk under the candidate dir.
	pinned, err := os.ReadFile(filepath.Join(dir, "list.csv"))
	require.NoError(t, err)
	require.Equal(t, body, string(pinned))

	// A second call finds the pinned file locally (server can be gone).
	dl2 := download.New(log.NewTestLogger(), nil)
	files2, err := dl2.GetFiles(context.Background(), dir, map[string]string{"list.csv": "http://127.0.0.1:1/unreachable"})
	require.NoError(t, err)
	t.Cleanup(func() { files2.Close() })
	require.Len(t, files2, 1)
}

// TestClient_GetFiles_DoesNotWriteUnmarkedDir verifies operator-provided data
// directories are never modified by fetched downloads.
func TestClient_GetFiles_DoesNotWriteUnmarkedDir(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, "fresh-bytes")
	}))
	t.Cleanup(srv.Close)

	dir := t.TempDir()
	dl := download.New(log.NewTestLogger(), nil)
	files, err := dl.GetFiles(context.Background(), dir, map[string]string{"list.csv": srv.URL})
	require.NoError(t, err)
	t.Cleanup(func() { files.Close() })
	require.Len(t, files, 1)

	_, err = os.Stat(filepath.Join(dir, "list.csv"))
	require.True(t, os.IsNotExist(err), "unmarked directory must not be written to")
}
