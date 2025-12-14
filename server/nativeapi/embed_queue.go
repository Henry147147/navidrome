package nativeapi

import (
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/google/uuid"
)

type embedJobStatus string

const (
	StatusQueued    embedJobStatus = "queued"
	StatusRunning   embedJobStatus = "running"
	StatusSucceeded embedJobStatus = "succeeded"
	StatusFailed    embedJobStatus = "failed"
)

type embedJob struct {
	ID          string         `json:"id"`
	Status      embedJobStatus `json:"status"`
	Error       string         `json:"error,omitempty"`
	EnqueuedAt  time.Time      `json:"enqueuedAt"`
	StartedAt   time.Time      `json:"startedAt,omitempty"`
	CompletedAt time.Time      `json:"completedAt,omitempty"`
	MusicName   string         `json:"musicName,omitempty"`
	Result      map[string]any `json:"result,omitempty"`
}

type embedWork struct {
	job    *embedJob
	run    func() error
	result map[string]any
	music  string
}

type embedQueue struct {
	mu     sync.RWMutex
	jobs   map[string]*embedJob
	workCh chan *embedWork
	stopCh chan struct{}
	wg     sync.WaitGroup
}

func newEmbedQueue() *embedQueue {
	q := &embedQueue{
		jobs:   make(map[string]*embedJob),
		workCh: make(chan *embedWork, 64),
		stopCh: make(chan struct{}),
	}
	q.wg.Add(1)
	go q.loop()
	return q
}

func (q *embedQueue) loop() {
	defer q.wg.Done()
	for {
		select {
		case work := <-q.workCh:
			if work == nil {
				continue
			}
			q.markRunning(work.job)
			err := work.run()
			if err != nil {
				q.markFailed(work.job, err)
				continue
			}
			q.markSucceeded(work.job, work.result)
		case <-q.stopCh:
			return
		}
	}
}

func (q *embedQueue) Stop() {
	close(q.stopCh)
	q.wg.Wait()
}

func (q *embedQueue) Enqueue(work *embedWork) (string, error) {
	if work == nil {
		return "", fmt.Errorf("work is nil")
	}

	jobID := uuid.NewString()
	job := &embedJob{
		ID:         jobID,
		Status:     StatusQueued,
		EnqueuedAt: time.Now(),
		MusicName:  work.music,
	}
	work.job = job

	q.mu.Lock()
	q.jobs[jobID] = job
	q.mu.Unlock()

	select {
	case q.workCh <- work:
		return jobID, nil
	default:
		q.mu.Lock()
		delete(q.jobs, jobID)
		q.mu.Unlock()
		return "", fmt.Errorf("embed queue is full")
	}
}

func (q *embedQueue) Get(jobID string) (*embedJob, bool) {
	q.mu.RLock()
	defer q.mu.RUnlock()
	job, ok := q.jobs[jobID]
	return job, ok
}

func (q *embedQueue) List(limit int) []*embedJob {
	q.mu.RLock()
	defer q.mu.RUnlock()

	jobs := make([]*embedJob, 0, len(q.jobs))
	for _, job := range q.jobs {
		jobs = append(jobs, job)
	}

	sort.Slice(jobs, func(i, j int) bool {
		return jobs[i].EnqueuedAt.After(jobs[j].EnqueuedAt)
	})

	if limit > 0 && len(jobs) > limit {
		return jobs[:limit]
	}
	return jobs
}

func (q *embedQueue) markRunning(job *embedJob) {
	q.mu.Lock()
	defer q.mu.Unlock()
	job.Status = StatusRunning
	job.StartedAt = time.Now()
}

func (q *embedQueue) markSucceeded(job *embedJob, result map[string]any) {
	q.mu.Lock()
	defer q.mu.Unlock()
	job.Status = StatusSucceeded
	job.CompletedAt = time.Now()
	job.Result = result
}

func (q *embedQueue) markFailed(job *embedJob, err error) {
	q.mu.Lock()
	defer q.mu.Unlock()
	job.Status = StatusFailed
	job.CompletedAt = time.Now()
	if err != nil {
		job.Error = err.Error()
	}
}
