use std::hash::Hash;

use hashbrown::HashSet;

pub struct Counter<K> {
    list: Vec<K>,
}

impl<K> Default for Counter<K> {
    fn default() -> Self {
        Self { list: Default::default() }
    }
}

impl<K: Eq + Hash> Counter<K> {
    pub fn inc(&mut self, key: K) {
        self.list.push(key)
    }

    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }

    /// The number of unique keys.
    pub fn distinct(&self) -> usize {
        self.list.iter().collect::<HashSet<&K>>().len()
    }

    pub fn get(&self, key: &K) -> usize {
        self.list.iter().filter(|&k| k == key).count()
    }

    /// Compute the sum of the counts.
    pub fn total(&self) -> usize {
        self.list.len()
    }
}
