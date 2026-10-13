use std::collections::HashMap;

pub struct Vocab {
    pieces: Vec<String>,
    special: Vec<bool>,
}

impl Vocab {
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let m: HashMap<String, String> = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        let n = m.keys().filter_map(|k| k.parse::<usize>().ok()).max().map_or(0, |x| x + 1);
        let mut pieces = vec![String::new(); n];
        for (k, v) in m {
            if let Ok(i) = k.parse::<usize>() {
                pieces[i] = v;
            }
        }
        let special = pieces
            .iter()
            .map(|p| (p.starts_with("<|") && p.ends_with("|>")) || p == "<unk>" || p == "<pad>")
            .collect();
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
