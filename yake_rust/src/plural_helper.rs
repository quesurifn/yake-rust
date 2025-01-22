pub trait PluralHelper {
    /// Omit the last `s` symbol in a string.
    ///
    /// How to use: `some_string.to_lowercase().to_single()`
    fn to_single(self) -> Self;
}

impl<'a> PluralHelper for &'a str {
    fn to_single(self) -> &'a str {
        if self.chars().count() > 3 && (self.ends_with(['s', 'S'])) {
            let mut chars = self.chars();
            chars.next_back();
            chars.as_str()
        } else {
            self
        }
    }
}
