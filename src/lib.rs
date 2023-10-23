#![warn(missing_docs)]
//! A simple PoC crate for splitting computations on large arrays between
//! threads with [`rayon`]

use std::ops::DerefMut;

use rayon::prelude::*;

/// After reaching this threshold computations will be parallelized
pub const THRESHOLD: usize = 64;

/// Runs `map` on each element of `src`, parallelizing when `src` is big enough.
///
/// If you want to transform a vector of data in-place, check out
/// [`threaded_mutate`].
///
/// # Example
/// ```
/// # use vemcap::threaded_map;
/// let input = vec!["123", "456", "789"];
/// let output: Vec<_> = threaded_map(input, |x| x.parse::<u16>().unwrap());
/// assert_eq!(output, vec![123, 456, 789]);
/// ```
pub fn threaded_map<T, F, U, R>(src: Vec<T>, map: F) -> R
where
	F: Fn(T) -> U + Send + Sync,
	R: FromIterator<U> + FromParallelIterator<U>,
	T: Send,
	U: Send,
{
	if src.len() < THRESHOLD {
		src.into_iter().map(map).collect()
	} else {
		src.into_par_iter().map(map).collect()
	}
}

/// A more efficient version of [`threaded_map`] for in-place data
/// transformation.
///
/// # Example
/// ```
/// # use vemcap::threaded_mutate;
/// let mut data = vec![1, 2, 3, 4];
/// threaded_mutate(&mut data, |x| *x *= *x);
/// assert_eq!(data, vec![1, 4, 9, 16]);
/// ```
pub fn threaded_mutate<S, T, F>(src: &mut S, map: F)
where
	S: DerefMut<Target = [T]>,
	F: Fn(&mut T) + Send + Sync,
	T: Send,
{
	if src.len() < THRESHOLD {
		src.iter_mut().for_each(map)
	} else {
		src.par_iter_mut().for_each(map)
	}
}

#[cfg(test)]
mod tests {
	mod threaded_map {
		use super::super::threaded_map;

		#[test]
		fn squares() {
			let input = (0..1024u32).collect();
			let expected: Vec<_> = (0..1024u32).map(|x| x.pow(2)).collect();
			let output: Vec<_> = threaded_map(input, |x| x.pow(2));
			assert_eq!(output, expected);
		}
	}

	mod threaded_mutate {
		use super::super::threaded_mutate;

		#[test]
		fn increment() {
			let mut data: Vec<_> = (0..1024u32).collect();
			let expected: Vec<_> = (1..=1024).collect();
			threaded_mutate(&mut data, |x| *x += 1);

			assert_eq!(data, expected);
		}
	}
}
