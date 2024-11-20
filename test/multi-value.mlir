module {
    func.func @multi_value(%arg0: i32, %arg1: i32) -> (i32, i32) {
        %0 = arith.addi %arg0, %arg0 : i32
        %1 = arith.addi %arg1, %arg1 : i32
        return %0, %1 : i32, i32
    }

    func.func @main(%arg0: i32) -> i32 {
        %c0 = arith.constant 1 : i32
        %c1 = arith.constant 2 : i32
        cf.br ^bb0(%c0, %c1 : i32, i32)
        ^bb0(%0: i32, %1: i32) : 
            %2, %3 = call @multi_value(%0, %1) : (i32, i32) -> (i32, i32)
            %4 = arith.addi %2, %3 : i32
            %5 = arith.constant 10 : i32
            %b = arith.cmpi sgt, %4, %5 : i32
            cf.cond_br %b, ^bb0(%2, %4 : i32, i32), ^bb1(%2 : i32)
        ^bb1(%r: i32) : 
            return %r : i32
    }
}