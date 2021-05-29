
use ndarray::{par_azip, s, Array, Array1, Array2, Array3, ArrayD, Ix3};

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr2, arr3};

    #[test]
    fn test_unfold1() {
        let input = arr3(&[[
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.],
        ]])
        .into_dyn();
        let output = unfold_3d_matrix(1, input, 3, true);
        assert_eq!(
            output,
            arr2(&[
                [1., 2., 3., 5., 6., 7., 9., 10., 11.],
                [2., 3., 4., 6., 7., 8., 10., 11., 12.],
                [5., 6., 7., 9., 10., 11., 13., 14., 15.],
                [6., 7., 8., 10., 11., 12., 14., 15., 16.]
            ])
        );
    }
    #[test]
    fn test_unfold2() {
        let input = arr3(&[[
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.],
        ]])
        .into_dyn();
        let output = unfold_3d_matrix(1, input, 2, true);
        assert_eq!(
            output,
            arr2(&[
                [1., 2., 5., 6.],
                [2., 3., 6., 7.],
                [3., 4., 7., 8.],
                [5., 6., 9., 10.],
                [6., 7., 10., 11.],
                [7., 8., 11., 12.],
                [9., 10., 13., 14.],
                [10., 11., 14., 15.],
                [11., 12., 15., 16.]
            ])
        );
    }

    #[test]
    fn test_padding1() {
        let input = arr2(&[
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.],
        ])
        .into_dyn();
        let output = add_padding(1, input);
        assert_eq!(
            output,
            arr2(&[
                [0., 0., 0., 0., 0., 0.],
                [0., 1., 2., 3., 4., 0.],
                [0., 5., 6., 7., 8., 0.],
                [0., 9., 10., 11., 12., 0.],
                [0., 13., 14., 15., 16., 0.],
                [0., 0., 0., 0., 0., 0.]
            ])
            .into_dyn()
        );
    }
    #[test]
    fn test_padding2() {
        let input = arr3(&[
            [[1., 2., 3.], [5., 6., 7.]],
            [[9., 10., 11.], [13., 14., 15.]],
        ])
        .into_dyn();
        let output = add_padding(1, input);
        assert_eq!(
            output,
            arr3(&[
                [
                    [0., 0., 0., 0., 0.],
                    [0., 1., 2., 3., 0.],
                    [0., 5., 6., 7., 0.],
                    [0., 0., 0., 0., 0.]
                ],
                [
                    [0., 0., 0., 0., 0.],
                    [0., 9., 10., 11., 0.],
                    [0., 13., 14., 15., 0.],
                    [0., 0., 0., 0., 0.]
                ]
            ])
            .into_dyn()
        );
    }
}

/// We create a new Array of zeros with the size of the original input+padding.
/// Afterwards we copy the original image over to the center of the new image.
/// TODO change to full/normal/...
pub fn add_padding(padding: usize, input: ArrayD<f32>) -> ArrayD<f32> {
    let n = input.ndim(); // 2d or 3d input?
    let shape: &[usize] = input.shape();
    let x = shape[n - 2] + 2 * padding; // calculate the new dim with padding
    let y = shape[n - 1] + 2 * padding; // calculate the new dim with padding

    if n > 4 {
        unimplemented!();
    }

    if n == 4 {