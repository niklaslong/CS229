use ndarray::prelude::*;
use ndarray_linalg::solve::Inverse;

fn main() {
    let x = array![[1., 2.], [1., 5.], [1., 7.], [1., 8.]];
    let y = array![1., 2., 3., 3.];
    let w = array![0., 0.];
    // let h = h(w, x.row(0).to_owned());
    // let res = batch_gd(w, x, y, 0.01);
    let res = normal_eq(x, y);
    dbg!(res);
}

// Structures needed for the naive implementation (iterative):
//
//  - Cost function
//  - Training set
//
// Structures needed for the calculation using the normal equation:
//
//  - Matrix Algebra
//  - Training set
//  - Least squares (LMS)

// w: 2x1
// x: 1x2
// w*x: 2x2
fn h(w: Array1<f64>, x_m: Array1<f64>) -> f64 {
    // ß_0 + (ß_1 * x_1)
    w.dot(&x_m)
}

fn cost(w: &Array1<f64>, x: &Array2<f64>, y: &Array1<f64>, j: usize) -> f64 {
    let sum = x.outer_iter().zip(y.iter()).fold(0., |sum, (row, y)| {
        sum + (h(w.clone(), row.to_owned()) - *y) * row[j]
    });

    // TODO: forgot to square the sum and divid by two (this is done externally as the cost
    // function not squared and not divided by 2 can be used as is in the batch_gd.
    sum
}

// w: weights
// x: values evaluated in prediction function
// y: actual output values for x values in prediction function
// a: learning rate, usually good rule of thumb to set at 0.01, 0.02, 0.04 and so on and
// empirically determine which is most efficient
fn batch_gd(mut w: Array1<f64>, x: Array2<f64>, y: Array1<f64>, a: f64) -> Array1<f64> {
    // Using indices to travers the weights is necessary as this prevents long lived borrows and
    // allows w to be mutated and passed to the cost function.
    for _n in 1..10000 {
        for j in 0..w.len() {
            let sum = cost(&w, &x, &y, j);
            w[j] -= a * sum;
        }
    }

    w
}

// The sum is calculated only for one row of the data instead of for all rows before updating the
// weights.
fn stochastic_gd(mut w: Array1<f64>, x: Array2<f64>, y: Array1<f64>, a: f64) -> Array1<f64> {
    for _n in 1..10000 {
        for j in 0..w.len() {}
    }

    unimplemented!()
}

// Normal equation calculation.
fn normal_eq(x: Array2<f64>, y: Array1<f64>) -> Array1<f64> {
    let x_t = x.clone().reversed_axes();
    let left = x_t.dot(&x).inv().unwrap();
    let right = x_t.dot(&y);
    left.dot(&right)
}
