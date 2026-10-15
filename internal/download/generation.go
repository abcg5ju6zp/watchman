// Copyright The Moov Authors
// Use of this source code is governed by an Apache License
// license that can be found in the LICENSE file.

package download

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/moov-io/base/log"
)

// Refresh generation lifecycle phases recorded in the candidate manifest.
const (
	phaseBuilding  = "building"
	phasePrepared  = "prepared"
	phasePublished = "published"
)

const (
	candidatesDirName   = "candidates"
	dataDirName         = "data"
	manifestFileName    = "manifest.json"
	currentDirName      = "current"
	currentMetaName     = "current.json"
	backupDirName       = "current.bak"
)

// manifestSource is the persisted per-source provenance of a generation.
type manifestSource struct {
	Name string `json:"name"`
	Hash string `json:"hash"`

	Fresh bool `json:"fresh"`
	Stale bool `json:"stale,omitempty"`

	PriorGenerationID string `json:"priorGenerationId,omitempty"`
	Error             string `json:"error,omitempty"`
}

// generationManifest is written atomically into a candidate directory.
type generationManifest struct {
	GenerationID      string           `json:"generationId"`
	PriorGenerationID string           `json:"priorGenerationId,omitempty"`
	Phase             string           `json:"phase"`
	EmbeddingsEnabled bool             `json:"embeddingsEnabled"`
	Sources           []manifestSource `json:"sources"`

	// Files lists every regular file (relative to the candidate data dir) that
	// was pinned for this generation. Recovery refuses to resume when any file
	// is missing so a resumed generation can never mix in a fresh download.
	Files []string `json:"files"`

	CreatedAt   time.Time  `json:"createdAt"`
	PreparedAt  *time.Time `json:"preparedAt,omitempty"`
	PublishedAt *time.Time `json:"publishedAt,omitempty"`
}

// generationStore owns the on-disk layout of isolated refresh generations:
//
//	root/
//	  current/                     # files of the last published generation
//	  current.json                 # metadata of the last published generation
//	  candidates/<generationID>/
//	    manifest.json
//	    data/                      # pinned source files for the generation
type generationStore struct {
	logger log.Logger

	root       string
	initialDir string
}

// NewGenerationStore prepares (but does not populate) the refresh state
// directory derived from conf. A non-nil store is returned even when the
// directory cannot be created; callers may run without persistence.
func NewGenerationStore(logger log.Logger, conf Config) (*generationStore, error) {
	root := stateRootDirectory(conf)
	if err := os.MkdirAll(root, 0o755); err != nil {
		return &generationStore{logger: logger, root: root, initialDir: expandInitialDir(initialDataDirectory(conf))},
			fmt.Errorf("creating refresh state directory %s: %w", root, err)
	}
	return &generationStore{
		logger:     logger,
		root:       root,
		initialDir: expandInitialDir(initialDataDirectory(conf)),
	}, nil
}

func stateRootDirectory(conf Config) string {
	if v := strings.TrimSpace(os.Getenv("REFRESH_STATE_DIRECTORY")); v != "" {
		return v
	}
	if v := strings.TrimSpace(conf.RefreshStateDirectory); v != "" {
		return v
	}
	if initial := strings.TrimSpace(initialDataDirectory(conf)); initial != "" {
		return filepath.Join(initial, ".watchman-refresh")
	}
	return filepath.Join(os.TempDir(), "watchman-refresh")
}

// newGenerationID returns a sortable, unique generation identifier.
func newGenerationID() string {
	var b [4]byte
	if _, err := rand.Read(b[:]); err != nil {
		// crypto/rand failing is not recoverable; fall back to time-based
		// uniqueness rather than panicking in library code.
		return fmt.Sprintf("gen-%d", time.Now().UnixNano())
	}
	return "gen-" + time.Now().UTC().Format("20060102T150405.000000") + "-" + hex.EncodeToString(b[:])
}

// GenerationWorkspace is one in-flight (or resuming) refresh generation.
type GenerationWorkspace struct {
	store *generationStore

	id       string
	priorID  string
	dir      string
	dataDir  string
	manifest generationManifest
	resumed  bool
	carried  bool // current/ files were copied for stale sources
}

func (w *GenerationWorkspace) ID() string {
	if w == nil {
		return ""
	}
	return w.id
}

func (w *GenerationWorkspace) DataDir() string {
	if w == nil {
		return ""
	}
	return w.dataDir
}

func (w *GenerationWorkspace) PriorGenerationID() string {
	if w == nil {
		return ""
	}
	return w.priorID
}

func (w *GenerationWorkspace) Resumed() bool {
	if w == nil {
		return false
	}
	return w.resumed
}

// Create opens a brand-new candidate directory, pins the configured source set
// and seeds it with files from the operator's initial data directory (hardlinks
// when possible, copies across devices). The manifest starts in phase "building"
// so a crash before publish makes recovery discard the candidate.
func (s *generationStore) Create(embeddingsEnabled bool) (*GenerationWorkspace, error) {
	id := newGenerationID()
	ws := &GenerationWorkspace{
		store:   s,
		id:      id,
		priorID: s.currentGenerationID(),
		dir:     filepath.Join(s.root, candidatesDirName, id),
		dataDir: filepath.Join(s.root, candidatesDirName, id, dataDirName),
	}
	ws.manifest = generationManifest{
		GenerationID:      id,
		PriorGenerationID: ws.priorID,
		Phase:             phaseBuilding,
		EmbeddingsEnabled: embeddingsEnabled,
		CreatedAt:         time.Now().UTC(),
	}

	if err := os.MkdirAll(ws.dataDir, 0o755); err != nil {
		return nil, fmt.Errorf("creating candidate directory: %w", err)
	}

	if err := copyTopLevelFiles(s.logger, s.initialDir, ws.dataDir); err != nil {
		s.logger.Warn().Logf("seeding candidate from %s failed: %v", s.initialDir, err)
	}

	if err := ws.writeManifest(); err != nil {
		ws.Discard()
		return nil, err
	}
	return ws, nil
}

// CarryCurrentFiles copies the last published generation's raw files into the
// candidate data directory so sources that fail this refresh (but are allowed
// to) can keep serving the previous generation, and so an interrupted publish
// can be rebuilt from pinned files after a restart. Best effort.
func (w *GenerationWorkspace) CarryCurrentFiles() error {
	if w == nil || w.carried {
		return nil
	}
	w.carried = true
	current := filepath.Join(w.store.root, currentDirName)
	return copyTopLevelFiles(w.store.logger, current, w.dataDir)
}

// MarkPrepared records the finished, not-yet-published generation: every source
// result and every pinned file. A crash after this returns (before publish)
// makes recovery resume this exact candidate.
func (w *GenerationWorkspace) MarkPrepared(stats Stats, embeddingsEnabled bool) error {
	if w == nil {
		return nil
	}

	files, err := listTopLevelFiles(w.dataDir)
	if err != nil {
		return err
	}

	names := make([]string, 0, len(stats.SourceResults))
	for name := range stats.SourceResults {
		names = append(names, name)
	}
	sort.Strings(names)

	sources := make([]manifestSource, 0, len(names))
	for _, name := range names {
		r := stats.SourceResults[name]
		sources = append(sources, manifestSource{
			Name:              r.Name,
			Hash:              r.Hash,
			Fresh:             r.Fresh,
			Stale:             r.Stale,
			PriorGenerationID: r.PriorGenerationID,
			Error:             r.Error,
		})
	}

	now := time.Now().UTC()
	w.manifest.Phase = phasePrepared
	w.manifest.EmbeddingsEnabled = embeddingsEnabled
	w.manifest.Sources = sources
	w.manifest.Files = files
	w.manifest.PreparedAt = &now

	return w.writeManifest()
}

// ApplyProvenance restores per-source provenance (stale markers and carried
// generation IDs) from a resumed manifest onto stats parsed from the pinned
// files. Download/parse cannot tell a carried-over file from a fresh one, so
// the manifest is the source of truth on resume.
func (w *GenerationWorkspace) ApplyProvenance(stats *Stats) {
	if w == nil || stats == nil || !w.resumed {
		return
	}
	for _, ms := range w.manifest.Sources {
		if !ms.Stale {
			continue
		}
		if stats.SourceResults == nil {
			stats.SourceResults = make(map[string]SourceResult)
		}
		cur := stats.SourceResults[ms.Name]
		cur.Name = ms.Name
		cur.Fresh = false
		cur.Stale = true
		cur.Hash = ms.Hash
		cur.PriorGenerationID = ms.PriorGenerationID
		cur.Entities = stats.Lists[ms.Name]
		stats.SourceResults[ms.Name] = cur
	}
	stats.GenerationID = w.id
}

// Matches verifies stats produced from a resumed candidate equal the pinned
// manifest: same source set, same per-source hashes. Any mismatch means the
// candidate can no longer be completed as one generation and must be discarded.
func (w *GenerationWorkspace) Matches(stats Stats) error {
	if w == nil {
		return errors.New("nil workspace")
	}
	if len(w.manifest.Sources) != len(stats.ListHashes) {
		return fmt.Errorf("generation %s source set changed: manifest has %d sources, reload found %d",
			w.id, len(w.manifest.Sources), len(stats.ListHashes))
	}
	for _, ms := range w.manifest.Sources {
		got, ok := stats.ListHashes[ms.Name]
		if !ok {
			return fmt.Errorf("generation %s missing source %s on resume", w.id, ms.Name)
		}
		if got != ms.Hash {
			return fmt.Errorf("generation %s source %s changed: pinned hash %s, reload hash %s",
				w.id, ms.Name, ms.Hash, got)
		}
	}
	return nil
}

// Publish promotes the candidate's pinned files to the current generation and
// marks it published. In-memory indexes have already been swapped by the
// caller before Publish runs.
func (w *GenerationWorkspace) Publish(stats Stats) error {
	if w == nil {
		return nil
	}

	publishedAt := time.Now().UTC()
	w.manifest.Phase = phasePublished
	w.manifest.PublishedAt = &publishedAt

	current := filepath.Join(w.store.root, currentDirName)
	backup := filepath.Join(w.store.root, backupDirName)

	// Swap data dirs in two renames (same filesystem). A crash between them is
	// repaired on recovery by rebuilding from the untouched candidate data dir.
	if err := os.RemoveAll(backup); err != nil {
		w.store.logger.Warn().Logf("removing stale %s: %v", backup, err)
	}
	if _, err := os.Stat(current); err == nil {
		if err := os.Rename(current, backup); err != nil {
			return fmt.Errorf("rotating current generation: %w", err)
		}
	}
	if err := os.Rename(w.dataDir, current); err != nil {
		// Put the previous generation back so the on-disk cache never disappears.
		if _, statErr := os.Stat(backup); statErr == nil {
			_ = os.Rename(backup, current)
		}
		return fmt.Errorf("promoting generation %s: %w", w.id, err)
	}
	if err := os.RemoveAll(backup); err != nil {
		w.store.logger.Warn().Logf("removing previous generation backup: %v", err)
	}

	if err := writeJSONAtomic(filepath.Join(w.store.root, currentMetaName), w.manifest); err != nil {
		w.store.logger.Warn().Logf("writing current generation metadata: %v", err)
	}

	return os.RemoveAll(w.dir)
}

// Discard removes every artifact of the candidate. It must be called when a
// refresh fails so a later generation can never read partial files.
func (w *GenerationWorkspace) Discard() {
	if w == nil {
		return
	}
	if err := os.RemoveAll(w.dir); err != nil {
		w.store.logger.Warn().Logf("discarding candidate %s: %v", w.id, err)
	}
}

func (w *GenerationWorkspace) writeManifest() error {
	return writeJSONAtomic(filepath.Join(w.dir, manifestFileName), w.manifest)
}

// currentGenerationID returns the ID of the last published generation, if any.
func (s *generationStore) currentGenerationID() string {
	var m generationManifest
	if err := readJSON(filepath.Join(s.root, currentMetaName), &m); err != nil {
		return ""
	}
	return m.GenerationID
}

// Recover inspects leftover candidates after a process exit. It completes at
// most one candidate (the newest fully-prepared one) by returning it for the
// startup refresh to resume; every other leftover is discarded. Incomplete,
// corrupt or publishing leftovers are removed before any new download starts.
func (s *generationStore) Recover() (*GenerationWorkspace, error) {
	candidatesRoot := filepath.Join(s.root, candidatesDirName)
	entries, err := os.ReadDir(candidatesRoot)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("reading candidates directory: %w", err)
	}

	var resumable []*GenerationWorkspace
	for _, entry := range entries {
		dir := filepath.Join(candidatesRoot, entry.Name())
		ws, reason := s.loadCandidate(dir)
		if ws == nil {
			s.logger.Warn().Logf("discarding interrupted refresh candidate %s: %s", entry.Name(), reason)
			if err := os.RemoveAll(dir); err != nil {
				s.logger.Warn().Logf("removing candidate %s: %v", dir, err)
			}
			continue
		}
		resumable = append(resumable, ws)
	}

	if len(resumable) == 0 {
		return nil, nil
	}

	// Newest prepared candidate wins; older leftovers are discarded.
	sort.Slice(resumable, func(i, j int) bool {
		return resumable[i].id > resumable[j].id
	})
	chosen := resumable[0]
	for _, ws := range resumable[1:] {
		ws.Discard()
	}
	chosen.resumed = true
	s.logger.Info().Logf("resuming interrupted refresh generation %s (%d pinned files)", chosen.id, len(chosen.manifest.Files))
	return chosen, nil
}

// loadCandidate validates one leftover candidate. Returns a workspace only for
// a fully-prepared candidate whose pinned files are all present.
func (s *generationStore) loadCandidate(dir string) (*GenerationWorkspace, string) {
	var m generationManifest
	if err := readJSON(filepath.Join(dir, manifestFileName), &m); err != nil {
		return nil, "missing or unreadable manifest: " + err.Error()
	}
	if m.Phase != phasePrepared {
		return nil, "generation was never prepared (phase=" + m.Phase + ")"
	}
	if m.GenerationID == "" {
		return nil, "manifest has no generation id"
	}

	dataDir := filepath.Join(dir, dataDirName)
	for _, rel := range m.Files {
		if filepath.IsAbs(rel) || strings.Contains(rel, "..") {
			return nil, "unsafe manifest file path: " + rel
		}
		info, err := os.Stat(filepath.Join(dataDir, rel))
		if err != nil {
			return nil, "pinned file missing: " + rel
		}
		if info.IsDir() || info.Size() == 0 {
			return nil, "pinned file invalid: " + rel
		}
	}

	return &GenerationWorkspace{
		store:    s,
		id:       m.GenerationID,
		priorID:  m.PriorGenerationID,
		dir:      dir,
		dataDir:  dataDir,
		manifest: m,
	}, ""
}

// copyTopLevelFiles links (or copies across devices) every regular file
// directly under src into dst. Existing destination files are kept.
func copyTopLevelFiles(logger log.Logger, src, dst string) error {
	if strings.TrimSpace(src) == "" {
		return nil
	}
	entries, err := os.ReadDir(src)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	if err := os.MkdirAll(dst, 0o755); err != nil {
		return err
	}
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || strings.HasPrefix(name, ".") {
			continue
		}
		srcPath := filepath.Join(src, name)
		dstPath := filepath.Join(dst, name)
		if _, err := os.Stat(dstPath); err == nil {
			continue // already pinned
		}
		if err := linkOrCopy(srcPath, dstPath); err != nil && logger != nil {
			logger.Warn().Logf("carrying %s into candidate: %v", name, err)
		}
	}
	return nil
}

func linkOrCopy(srcPath, dstPath string) error {
	if err := os.Link(srcPath, dstPath); err == nil {
		return nil
	}
	in, err := os.Open(srcPath)
	if err != nil {
		return err
	}
	defer in.Close()

	info, err := in.Stat()
	if err != nil {
		return err
	}
	out, err := os.OpenFile(dstPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, info.Mode().Perm())
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		return err
	}
	return out.Close()
}

func listTopLevelFiles(dir string) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var files []string
	for _, entry := range entries {
		if entry.IsDir() || strings.HasPrefix(entry.Name(), ".") {
			continue
		}
		files = append(files, entry.Name())
	}
	sort.Strings(files)
	return files, nil
}

func writeJSONAtomic(path string, v any) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".manifest-*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	enc := json.NewEncoder(tmp)
	enc.SetIndent("", "  ")
	if err := enc.Encode(v); err != nil {
		tmp.Close()
		os.Remove(tmpName)
		return err
	}
	if err := tmp.Sync(); err != nil {
		// fsync best effort
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpName)
		return err
	}
	if err := os.Rename(tmpName, path); err != nil {
		os.Remove(tmpName)
		return err
	}
	return nil
}

func readJSON(path string, v any) error {
	fd, err := os.Open(path)
	if err != nil {
		return err
	}
	defer fd.Close()
	return json.NewDecoder(fd).Decode(v)
}

// manifestSourceNames is a small test/debug helper.
func manifestSourceNames(m *generationManifest) []string {
	if m == nil {
		return nil
	}
	out := make([]string, 0, len(m.Sources))
	for _, src := range m.Sources {
		out = append(out, src.Name)
	}
	sort.Strings(out)
	return slices.Compact(out)
}
