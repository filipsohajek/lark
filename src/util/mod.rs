pub mod file;

use std::cell::RefCell;
use std::collections::HashMap;
use memmap::MmapMut;

/// Represents a string interned in the thread local string interner.
/// This is a convenience type for interfacing with the interner that allows interned strings
/// to be treated similarly to other string types.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct InternedStr(usize);

impl InternedStr {
    /// Construct a new interned string from a given existing string. A copy of the string is made
    /// if not yet present in the interner.
    pub fn new_from_str(string: &str) -> Self {
        let str_id = STRING_INTERNER.with(|interner_ref| {
            interner_ref.borrow_mut().intern(string)
        });

        Self(str_id)
    }

    /// Check if the interned string matches a provided non-interned string.
    ///
    /// Interned strings can be freely compared with each other through the PartialEq implementation.
    pub fn matches(&self, other: &str) -> bool {
        STRING_INTERNER.with(|interner_ref| {
            interner_ref.borrow().lookup(self.0).unwrap() == other
        })
    }

    /// Return a reference to the backing string in the interner.
    ///
    /// As the interner is append-only and static, the resulting reference has a static lifetime.
    pub fn to_str(&self) -> &'static str {
        STRING_INTERNER.with(|interner_ref| {
            interner_ref.borrow().lookup(self.0).unwrap()
        })
    }
}


/// An interner for strings (not thread-safe). Used to store identifiers encountered during lexing.
///
/// The interner stores a vector of memory-mapped blocks, in which the individual strings are
/// stored, each one in a contiguous chunk of one of the blocks. The allocation procedure is very
/// simple, always using the topmost block for new allocations and creating a larger block if the
/// remaining capacity is not sufficient to fulfill an allocation request. This would be inefficient
/// in general, but works reasonably well for identifier names, which are much smaller than the
/// default block size (512 bytes).
///
/// The strings are indexed by an ID of type usize and the interner keeps a vector lookup table from
/// these IDs to the strings, which allows for constant-time lookup. A similar hashtable-based index
/// is kept, mapping existing strings to their IDs, making the insertion of an already existing
/// string an O(1) operation.
///
/// The interner bypasses the borrow checker and issues static-lifetime references to the interned
/// strings. Care must be taken so that the interner survives all its issued interned strings. Thus
/// it is not advisable to use the interner directly and to use the InternedStr API instead,
/// which uses a thread-local interner.
struct StrInterner {
    blocks: Vec<MmapMut>,
    intern_index: HashMap<&'static str, usize>,
    lookup_index: Vec<&'static str>,
    remaining_top: usize
}

impl StrInterner {
    const BLOCK_SIZE: usize = 4096;

    fn new() -> Self {
        Self {
            blocks: vec![],
            intern_index: HashMap::new(),
            lookup_index: vec![],
            remaining_top: 0,
        }
    }

    /// Ensure that there is enough space in the topmost block to store the provided amount of bytes
    /// and return a reference to the start of this contiguous free space.
    ///
    /// If the topmost block doesn't have enough space, a new one is created with its size exactly
    /// matching the allocation request.
    unsafe fn ensure_space(&mut self, length: usize) -> &'static mut [u8] {
        if length > self.remaining_top {
            let new_block_size = Self::BLOCK_SIZE.max(length);
            let new_block = MmapMut::map_anon(new_block_size).unwrap();
            self.blocks.push(new_block);
            self.remaining_top = new_block_size;
        }

        let top_block = self.blocks.last_mut().unwrap();
        let start_index = top_block.len() - self.remaining_top;

        std::mem::transmute::<&mut [u8], &mut [u8]>(&mut top_block[start_index..])
    }

    /// Add a string to the interner. Does not check if the string is already present.
    ///
    /// Allocates the required memory, copies the string into its block and updates both the
    /// lookup and interning index.
    fn add_string(&mut self, string: &str) -> usize {
        let str_space = unsafe {self.ensure_space(string.len())};
        let target_slice = &mut str_space[0..string.len()];
        target_slice.copy_from_slice(string.as_bytes());

        unsafe {
            let str_ref = std::str::from_utf8_unchecked(target_slice);
            self.lookup_index.push(str_ref);
            self.intern_index.insert(str_ref, self.lookup_index.len() - 1);
        }

        self.lookup_index.len() - 1
    }

    /// Intern a string.
    ///
    /// If the string is already present in the interner, returns its index. Otherwise it adds
    /// the string using the add_string method.
    fn intern(&mut self, to_intern: &str) -> usize {
        match self.intern_index.get(to_intern) {
            Some(existing_index) => *existing_index,
            None => self.add_string(to_intern)
        }
    }

    /// Try to lookup a string in the interner by its ID and return its reference if present.
    ///
    /// The returned reference has a static lifetime, which is not ensured by the borrow checker,
    /// but rather by the interner itself. Care must be taken to keep the interner alive throughout
    /// the program lifetime (or at least throughout the use of its interned strings).
    fn lookup(&self, str_id: usize) -> Option<&'static str> {
        self.lookup_index.get(str_id).map(|val| *val)
    }
}

thread_local! {
    static STRING_INTERNER: RefCell<StrInterner> = RefCell::new(StrInterner::new());
}
