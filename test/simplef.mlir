module {
    func.func @foo(%i: f32) -> (f32) {
	%a = arith.constant 2.0 : f32
	%b = arith.constant 4.0 : f32
	%c = arith.addf %a, %b : f32
	%d = arith.addf %c, %i : f32
	%e = arith.constant 10.0 : f32
	%f = arith.mulf %d, %e : f32
	return %f : f32
    }

}

