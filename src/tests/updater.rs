use self_update::version::bump_is_greater;

#[test]
fn newer_version_detected() {
    assert!(bump_is_greater("0.1.0", "0.2.0").unwrap());
    assert!(!bump_is_greater("0.2.0", "0.1.0").unwrap());
    assert!(!bump_is_greater("0.1.0", "0.1.0").unwrap());
}
