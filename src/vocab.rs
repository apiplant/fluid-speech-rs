use std::collections::HashMap;

pub struct Vocab {
    pieces: Vec<String>,
    special: Vec<bool>,
}

impl Vocab {
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let (pieces, n) = if path.is_file() {
            let m: HashMap<String, String> = serde_json::from_str(&std::fs::read_to_string(path)?)?;
            let n = m.keys().filter_map(|k| k.parse::<usize>().ok()).max().map_or(0, |x| x + 1);
            let mut pieces = vec![String::new(); n];
            for (k, v) in m {
                if let Ok(i) = k.parse::<usize>() {
                    pieces[i] = v;
                }
            }
            (pieces, n)
        } else {
            // Transformers stores the Parakeet token -> ID mapping in tokenizer.json's BPE model.
            let tokenizer_path = path.with_file_name("tokenizer.json");
            let json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&tokenizer_path)?)?;
            let vocab = json.get("model").and_then(|m| m.get("vocab")).and_then(serde_json::Value::as_object)
                .ok_or_else(|| anyhow::anyhow!("{} has no tokenizer model vocab", tokenizer_path.display()))?;
            let n = vocab.values().filter_map(serde_json::Value::as_u64).max().map_or(0, |x| x as usize + 1);
            let mut pieces = vec![String::new(); n];
            for (piece, id) in vocab {
                if let Some(id) = id.as_u64() {
                    pieces[id as usize] = piece.clone();
                }
            }
            (pieces, n)
        };
        let special = pieces
            .iter()
            .map(|p| (p.starts_with("<|") && p.ends_with("|>")) || p == "<unk>" || p == "<pad>" || p == "<blank>")
            .collect();
        debug_assert_eq!(pieces.len(), n);
        Ok(Self { pieces, special })
    }

    pub fn is_special(&self, id: u32) -> bool {
        self.special.get(id as usize).copied().unwrap_or(true)
    }

    pub fn piece(&self, id: u32) -> &str {
        self.pieces.get(id as usize).map(String::as_str).unwrap_or("")
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut s = String::new();
        for &i in ids {
            if self.special.get(i as usize).copied().unwrap_or(true) {
                continue;
            }
            s.push_str(&self.pieces[i as usize]);
        }
        s.replace('\u{2581}', " ").trim().to_string()
    }
}
