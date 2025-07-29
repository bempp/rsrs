use rlst::prelude::*;
use std::rc::Rc;
pub struct NormalOperator<
    'a,
    Space: IndexableSpace,
    Op: OperatorBase<Domain = Space, Range = Space>,
> {
    pub op: &'a Op,
}

impl<'a, Space: IndexableSpace, Op: OperatorBase<Domain = Space, Range = Space>> std::fmt::Debug
    for NormalOperator<'a, Space, Op>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dim_1 = self.op.domain().dimension();
        let dim_2 = self.op.range().dimension();
        write!(f, "Id Operator: [{dim_1}x{dim_2}]").unwrap();
        Ok(())
    }
}

impl<'a, Space: IndexableSpace, Op: OperatorBase<Domain = Space, Range = Space>> OperatorBase
    for NormalOperator<'a, Space, Op>
{
    type Domain = Space;
    type Range = Space;

    fn domain(&self) -> Rc<Self::Domain> {
        self.op.domain().clone()
    }

    fn range(&self) -> Rc<Self::Range> {
        self.op.range().clone()
    }
}

impl<'a, Space: IndexableSpace, Op: AsApply<Domain = Space, Range = Space>> AsApply
    for NormalOperator<'a, Space, Op>
{
    fn apply_extended<
        ContainerIn: ElementContainer<E = <Self::Domain as LinearSpace>::E>,
        ContainerOut: ElementContainerMut<E = <Self::Range as LinearSpace>::E>,
    >(
        &self,
        alpha: <Self::Range as LinearSpace>::F,
        x: Element<ContainerIn>,
        beta: <Self::Range as LinearSpace>::F,
        y: Element<ContainerOut>,
        _trans_mode: TransMode,
    ) {
        self.op.apply_extended(
            alpha,
            self.op.apply(x, TransMode::NoTrans),
            beta,
            y,
            TransMode::ConjTrans,
        );
    }

    fn apply<ContainerIn: ElementContainer<E = <Self::Domain as LinearSpace>::E>>(
        &self,
        x: Element<ContainerIn>,
        trans_mode: rlst::TransMode,
    ) -> rlst::operator::ElementType<<Self::Range as LinearSpace>::E> {
        let mut y = zero_element(self.range());
        self.apply_extended(
            <<Self::Range as LinearSpace>::F as num::One>::one(),
            x,
            <<Self::Range as LinearSpace>::F as num::Zero>::zero(),
            y.r_mut(),
            trans_mode,
        );
        y
    }
}
