module {
    func.func @addone(%x: i32) -> i32 {
        %c1 = arith.constant 1 : i32
        %temp = arith.addi %x, %c1 : i32
        %result = arith.addi %temp, %c1 : i32
        return %result : i32
    }
}
