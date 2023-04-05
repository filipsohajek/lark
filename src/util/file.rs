use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::path::{Path, PathBuf};
use memmap::MmapMut;

/// Represents a slice of a file in the thread local SourceMap. The type is intentionally small
/// (2x usize), as it's used in many places throughout the code.
///
/// A valid span lies entirely inside of exactly one mapped file. Methods on Spans may not
/// necessarily preserve the validity of a Span (i.e. may make one Span span multiple files).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Span {
    pub offset: usize,
    pub length: usize
}

impl Span {
    /// Load a new file into the thread-local SourceMap and return a Span spanning the entire file.
    pub fn new_from_file<T: AsRef<Path>>(path: T) -> std::io::Result<Span> {
        SOURCE_MAP.with(|sourcemap_ref| {
            sourcemap_ref.borrow_mut().load_file(path)
        })
    }

    /// Return a reference to the buffer the Span is referring to. Panics if the Span is invalid.
    pub fn buffer(&self) -> &'static str {
        SOURCE_MAP.with(|sourcemap_ref| {
            sourcemap_ref.borrow().buffer(self.offset, self.length)
        })
    }

    /// Return a subspan starting at the specified offset of the parent Span and of a specified length.
    ///
    /// If the offset is greater than the span length, a Span of zero length is returned, starting
    /// at the end of the parent Span.
    /// If the length is greater than the parent Span length minus the substr offset, the method
    /// returns a subspan of maximum possible length which starts at the specified offset.
    pub fn substr(self, sub_offset: usize, sub_length: usize) -> Span {
        let clamped_offset = sub_offset.min(self.length);
        let clamped_length = sub_length.min(self.length - clamped_offset);

        Span {
            offset: self.offset + clamped_offset,
            length: clamped_length,
        }
    }

    /// Returns a subspan without a specified length of bytes at the beginning. It is equivalent
    /// to calling substr with the maximal possible length.
    pub fn truncate_head(self, trunc_len: usize) -> Span {
        let new_len = self.length - trunc_len;
        self.substr(trunc_len, new_len)
    }
}

/// Holds all opened source files mapped in memory.
///
/// For addressing the files, a global offset is used, behaving as if all opened files were stored
/// sequentially in memory. This has limitations, however, as the SourceMap cannot distribute
/// buffers which span files, and thus the file boundaries may require special care when handling
/// the global offset.
///
/// The preferred way to interface with a SourceMap is to use the Span API, which uses a static
/// thread-local SourceMap.
#[derive(Debug)]
struct SourceMap {
    files: Vec<MmapMut>,
    file_index: HashMap<PathBuf, usize>
}

impl SourceMap {
    fn new() -> Self {
        Self {
            files: vec![],
            file_index: HashMap::new()
        }
    }

    /// Return the buffer starting at a given global offset of a specified length.
    ///
    /// The requested "slice" must lie entirely inside of exactly one mapped file.
    fn buffer(&self, offset: usize, length: usize) -> &'static str {
        // Find the first file whose end lies after the offset
        let (matching_offset, matching_file) =
            self.files
                .iter()
                .scan(0, |total_offset, file| {
                    *total_offset += file.len();
                    Some((*total_offset, file))
                })
                .find(|&(total_offset, _file)| {
                    total_offset >= offset
                })
                .unwrap();

        // We do not support creating buffers spanning multiple files
        assert!(length <= matching_file.len(), "attempted to create a buffer spanning multiple files");

        // Compute the intra-file offset from the global offset and slice the map appropriately
        let file_base_offset = matching_file.len() - (matching_offset - offset);
        let buffer_slice = &matching_file[file_base_offset..(file_base_offset + length)];

        unsafe {
            std::mem::transmute::<&str, &str>(std::str::from_utf8_unchecked(buffer_slice))
        }
    }

    /// Try to map a file into the file map and return the resulting file ID (index into the file
    /// map).
    ///
    /// If the file is already mapped, the method does not map the file anew, returning the existing
    /// mapping instead.
    fn map_file<T: AsRef<Path>>(&mut self, file_path: T) -> std::io::Result<usize> {
        if self.file_index.contains_key(file_path.as_ref()) {
            let file_id = self.file_index[file_path.as_ref()];
            return Ok(file_id);
        }

        let file =
            OpenOptions::new()
                .write(true)
                .create(true)
                .read(true)
                .open(file_path.as_ref())?;
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        self.files.push(mmap);
        self.file_index.insert(file_path.as_ref().to_path_buf(), self.files.len() - 1);

        Ok(self.files.len() - 1)
    }

    /// Load a file into the SourceMap and return the resulting Span corresponding to the entire file.
    fn load_file<T: AsRef<Path>>(&mut self, file_path: T) -> std::io::Result<Span> {
        let file_id = self.map_file(file_path)?;

        Ok(Span {
            offset: self.files
                        .iter()
                        .take(file_id + 1)
                        .map(|file_mmap| file_mmap.len())
                        .sum(),
            length: self.files[file_id].len(),
        })
    }
}

thread_local! {
    static SOURCE_MAP: RefCell<SourceMap> = RefCell::new(SourceMap::new());
}