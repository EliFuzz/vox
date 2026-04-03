use std::path::Path;

pub(super) const SPACE_REPLACEMENT: &[u8] = "▁".as_bytes();

pub(super) struct BinTokenizer {
    tokens_to_bytes: Vec<Vec<u8>>,
    space_string: &'static [u8],
}

impl BinTokenizer {
    pub(super) fn load(path: &Path) -> Self {
        let data = std::fs::read(path)
            .unwrap_or_else(|_| panic!("failed to read tokenizer file: {}", path.display()));
        let tokens_to_bytes = parse_tokenizer_data(&data);
        assert!(
            !tokens_to_bytes.is_empty(),
            "empty tokenizer file: {}",
            path.display()
        );
        Self {
            tokens_to_bytes,
            space_string: SPACE_REPLACEMENT,
        }
    }

    pub(super) fn decode(&self, tokens: &[i64]) -> String {
        let mut result_bytes = Vec::with_capacity(tokens.len() * 4);
        for &token in tokens {
            let idx = token as usize;
            if self.tokens_to_bytes.get(idx).is_none_or(|b| {
                b.is_empty() || (b.len() > 2 && b[0] == b'<' && b[b.len() - 1] == b'>')
            }) {
                continue;
            }
            result_bytes.extend_from_slice(&self.tokens_to_bytes[idx]);
        }
        let text = String::from_utf8_lossy(&result_bytes).into_owned();
        let space_str = std::str::from_utf8(self.space_string).unwrap_or("_");
        text.replace(space_str, " ").trim().to_string()
    }
}

fn parse_tokenizer_data(data: &[u8]) -> Vec<Vec<u8>> {
    let mut tokens = Vec::new();
    let mut offset = 0;
    while offset < data.len() {
        let first_byte = data[offset];
        offset += 1;
        if first_byte == 0 {
            tokens.push(Vec::new());
            continue;
        }
        let mut byte_count = 0usize;
        if first_byte < 128 {
            byte_count = first_byte as usize;
        }
        if first_byte >= 128 {
            if offset >= data.len() {
                break;
            }
            let second_byte = data[offset];
            offset += 1;
            byte_count = (second_byte as usize * 128) + first_byte as usize - 128;
        }
        if offset + byte_count > data.len() {
            break;
        }
        tokens.push(data[offset..offset + byte_count].to_vec());
        offset += byte_count;
    }
    tokens
}
