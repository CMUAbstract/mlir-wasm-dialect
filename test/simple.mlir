module {
    func.func @foo(%i: i32) {
	%a = arith.constant 2 : i32
	%b = arith.constant 4 : i32
	%c = arith.addi %a, %b : i32
	%d = arith.addi %c, %i : i32
    return 
    }

}

