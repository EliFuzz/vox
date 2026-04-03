pub fn normalize_with_punctuation(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let mut result = String::with_capacity(trimmed.len());
    let mut remaining = trimmed;
    while !remaining.is_empty() {
        let end = remaining
            .find(|c: char| matches!(c, '.' | ',' | '!' | '?' | ';' | ':'))
            .map(|i| i + 1);
        let (segment, rest) = end
            .map(|pos| (&remaining[..pos], &remaining[pos..]))
            .unwrap_or((remaining, ""));
        let seg_trimmed = segment.trim();
        if seg_trimmed.is_empty() {
            remaining = rest.trim_start();
            continue;
        }
        let last_byte = seg_trimmed.as_bytes()[seg_trimmed.len() - 1];
        let has_trailing = matches!(last_byte, b'.' | b',' | b'!' | b'?' | b';' | b':');
        if !has_trailing {
            let normalized = text_processing_rs::normalize_sentence(seg_trimmed);
            if !result.is_empty() && !result.ends_with(' ') {
                result.push(' ');
            }
            result.push_str(normalized.trim());
            remaining = rest.trim_start();
            continue;
        }
        let punct = &seg_trimmed[seg_trimmed.len() - 1..];
        let body = seg_trimmed[..seg_trimmed.len() - 1].trim();
        if body.is_empty() {
            result.push_str(punct);
            remaining = rest.trim_start();
            continue;
        }
        let normalized = text_processing_rs::normalize_sentence(body);
        if !result.is_empty() && !result.ends_with(' ') {
            result.push(' ');
        }
        result.push_str(normalized.trim());
        result.push_str(punct);
        remaining = rest.trim_start();
    }
    result
}
