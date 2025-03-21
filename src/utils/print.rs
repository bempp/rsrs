use rlst::{DynamicArray, RandomAccessByValue, RlstScalar, Shape};

pub fn pretty_print_advanced<Item: RlstScalar>(
    arr: &DynamicArray<Item, 2>,
    rows: usize,
    cols: usize,
    print_width: usize,
    mantissa: usize,
    exponent: usize,
) {
    let shape = (
        std::cmp::min(arr.shape()[0], rows),
        std::cmp::min(arr.shape()[1], cols),
    );
    let mut content_str = String::new();

    // For alignment in columns, must satisfy: 4 + mantissa + exponent < print_width
    let print_width = std::cmp::max(print_width, 5 + mantissa + exponent);

    let num_outer_spaces = print_width - mantissa - exponent - 4;

    let mut x_ij = arr.get_value([0, 0]).unwrap();
    for row in 0..shape.0 {
        content_str += "│";
        for col in 0..shape.1 {
            x_ij = arr.get_value([row, col]).unwrap();

            content_str += &format!(" {}", fmt_real(x_ij, print_width, mantissa, exponent));
        }
        content_str += &" ".repeat(num_outer_spaces);
        content_str += "│\n";
    }

    let colwidth = fmt_real(x_ij, print_width, mantissa, exponent)
        .chars()
        .count();
    let top_str = format!(
        "\n┌{}┐\n",
        " ".repeat((shape.1 * (colwidth + 1)) + num_outer_spaces)
    );
    let btm_str = format!(
        "└{}┘\n",
        " ".repeat((shape.1 * (colwidth + 1)) + num_outer_spaces)
    );
    println!(
        "Printing the upper left {} x {} block of a matrix with dimensions {} x {}.",
        shape.0,
        shape.1,
        arr.shape()[0],
        arr.shape()[1]
    );
    println!("{top_str}{content_str}{btm_str}");
}

pub fn pretty_print<Item: RlstScalar>(arr: &DynamicArray<Item, 2>) {
    pretty_print_advanced(arr, 10, 10, 11, 3, 2);
}

fn fmt_real<T: RlstScalar>(num: T, width: usize, precision: usize, exp_pad: usize) -> String {
    let mut num = format!("{:.precision$e}", num, precision = precision);
    // Safe to `unwrap` as `num` is guaranteed to contain `'e'`
    let exp = num.split_off(num.find('e').unwrap());

    let (sign, exp) = exp
        .strip_prefix("e-")
        .map_or_else(|| ('+', &exp[1..]), |stripped| ('-', stripped));

    num.push_str(&format!("e{}{:0>pad$}", sign, exp, pad = exp_pad));

    format!("{:>width$}", num, width = width)
}
