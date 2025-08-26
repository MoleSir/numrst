use num_traits::Zero;

// ---------- 分类与数值 trait ----------
pub trait NumCategory {}
pub struct IntCategory;
pub struct FloatCategory;
impl NumCategory for IntCategory {}
impl NumCategory for FloatCategory {}

pub trait Num: Copy + Sized + Zero {
    type Category: NumCategory;
}
pub trait FloatNum: Num<Category = FloatCategory> {}
pub trait IntNum:   Num<Category = IntCategory>   {}

// 示例：给基础类型挂上你的本地 trait
impl Num for f64 { type Category = FloatCategory; }
impl FloatNum for f64 {}

impl Num for i32 { type Category = IntCategory; }
impl IntNum for i32 {}

// ---------- 原始目标 trait ----------
pub trait ReduceOp<T: Num>: Sized {
    type Output: Num;
    fn op(arr: &[T]) -> Self::Output;
}

// ---------- 包装 trait：把类别提升为泛型参数 ----------
pub trait ReduceOpByCategory<T: Num, C: NumCategory = <T as Num>::Category> {
    type Output: Num;
    fn op(arr: &[T]) -> Self::Output;
}

// ---------- 实现们 ----------
struct ReduceSum;
impl<T: Num> ReduceOp<T> for ReduceSum {
    type Output = T;
    fn op(_arr: &[T]) -> Self::Output {
        println!("ReduceSum");
        T::zero()
    }
}

struct ReduceMean;

// 按类别分别实现；现在不会重叠
impl<T: FloatNum> ReduceOpByCategory<T, FloatCategory> for ReduceMean {
    type Output = T;
    fn op(arr: &[T]) -> Self::Output {
        println!("ReduceMean / Float");
        arr[0]
    }
}

impl<T: IntNum> ReduceOpByCategory<T, IntCategory> for ReduceMean {
    type Output = f64; // 注意需要让 f64 实现你的 Num（上面已实现）
    fn op(_arr: &[T]) -> Self::Output {
        println!("ReduceMean / Int");
        0.0
    }
}

// 统一转发：对外仍然是 ReduceOp<T>
impl<T: Num> ReduceOp<T> for ReduceMean
where
    ReduceMean: ReduceOpByCategory<T>, // 默认用 <T as Num>::Category
{
    type Output = <ReduceMean as ReduceOpByCategory<T>>::Output;

    fn op(arr: &[T]) -> Self::Output {
        <ReduceMean as ReduceOpByCategory<T>>::op(arr)
    }
}

// ---------- 小示例 ----------
fn main() {
    let ai = [1i32, 2, 3];
    let _ri: f64 = <ReduceMean as ReduceOp<i32>>::op(&ai);

    let af = [1.0f64, 2.0, 3.0];
    let _rf: f64 = <ReduceMean as ReduceOp<f64>>::op(&af); // Output = T = f64
}
