//! Append-only JSONL trial journal.
//!
//! Every state change to a `Trial` is recorded as one JSON line. Replaying
//! the file rebuilds the in-memory history (last write per `TrialId` wins),
//! so a crashed run can pick up where it left off by:
//! ```ignore
//! let history = TrialJournal::replay(&path)?;
//! let next_id = history.iter().map(|t| t.id.0).max().unwrap_or(0) + 1;
//! ```

use std::{
    fs::OpenOptions,
    io::{BufRead, BufReader, Seek, Write},
    path::{Path, PathBuf},
};

use crate::{error::TuneError, trial::Trial};

/// JSONL-backed trial recorder. Append-only; one line per record.
///
/// The file is opened in append mode on construction and held open for the
/// lifetime of the journal. `record` calls `flush()` after every write to
/// minimize the loss window if the process dies.
pub struct TrialJournal {
    path: PathBuf,
    file: std::fs::File,
}

impl TrialJournal {
    /// Create or open the journal at `path`. Parent directories must exist.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, TuneError> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(&path)?;
        Ok(Self { path, file })
    }

    /// Append one trial record to the journal. Always flushes.
    pub fn record(&mut self, trial: &Trial) -> Result<(), TuneError> {
        let mut line = serde_json::to_string(trial)?;
        line.push('\n');
        self.file.write_all(line.as_bytes())?;
        self.file.flush()?;
        Ok(())
    }

    /// Replay every record in `path` and return the consolidated history.
    /// If a `TrialId` appears multiple times (e.g. Running → Completed),
    /// the latest line wins.
    pub fn replay(path: impl AsRef<Path>) -> Result<Vec<Trial>, TuneError> {
        let file = match OpenOptions::new().read(true).open(path.as_ref()) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e.into()),
        };
        let reader = BufReader::new(file);
        // We can't use a HashMap<TrialId, Trial> directly without losing
        // insertion order, so collect into a Vec and dedupe last-wins.
        let mut by_id: std::collections::BTreeMap<u64, Trial> = std::collections::BTreeMap::new();
        for (lineno, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let trial: Trial = serde_json::from_str(&line).map_err(|e| {
                tracing::error!(line = lineno + 1, "journal parse error: {e}");
                TuneError::Serde(e)
            })?;
            by_id.insert(trial.id.0, trial);
        }
        Ok(by_id.into_values().collect())
    }

    /// Re-open the journal pointer; useful in tests that round-trip.
    pub fn reopen(&mut self) -> Result<(), TuneError> {
        self.file = OpenOptions::new()
            .append(true)
            .read(true)
            .open(&self.path)?;
        let _ = self.file.seek(std::io::SeekFrom::End(0))?;
        Ok(())
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use super::*;
    use crate::trial::{TrialId, TrialStatus};
    use serde_json::json;
    use std::collections::HashMap;

    #[test]
    fn journal_replay_reconstructs_history() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trials.jsonl");
        {
            let mut j = TrialJournal::open(&path).unwrap();
            for i in 0..5 {
                let mut t = Trial::new(
                    TrialId(i),
                    HashMap::from([("lr".to_string(), json!(0.001 * (i + 1) as f64))]),
                );
                j.record(&t).unwrap(); // running
                t.complete(0.5 - i as f64 * 0.05);
                j.record(&t).unwrap(); // completed (overwrites the running entry)
            }
        }
        let replayed = TrialJournal::replay(&path).unwrap();
        assert_eq!(replayed.len(), 5);
        for (i, t) in replayed.iter().enumerate() {
            assert_eq!(t.id, TrialId(i as u64));
            assert!(matches!(t.status, TrialStatus::Completed));
            assert!(t.metric.is_some());
        }
    }

    #[test]
    fn journal_replay_missing_file_yields_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("does-not-exist.jsonl");
        let replayed = TrialJournal::replay(&path).unwrap();
        assert!(replayed.is_empty());
    }
}
