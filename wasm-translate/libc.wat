  ;; unoptimized wat for malloc and free
(global $__stack_pointer (mut i32) (i32.const 148784))
(func $abort (type 3)
unreachable
unreachable)
(func $sbrk (type 0) (param i32) (result i32)
block  ;; label = @1
    local.get 0
    br_if 0 (;@1;)
    memory.size
    i32.const 16
    i32.shl
    return
end
block  ;; label = @1
    local.get 0
    i32.const 65535
    i32.and
    br_if 0 (;@1;)
    local.get 0
    i32.const -1
    i32.le_s
    br_if 0 (;@1;)
    block  ;; label = @2
    local.get 0
    i32.const 16
    i32.shr_u
    memory.grow
    local.tee 0
    i32.const -1
    i32.ne
    br_if 0 (;@2;)
    i32.const 0
    i32.const 48
    i32.store offset=82736
    i32.const -1
    return
    end
    local.get 0
    i32.const 16
    i32.shl
    return
end
call $abort
unreachable)
(func $malloc (type 0) (param i32) (result i32)
local.get 0
call $dlmalloc)
(func $dlmalloc (type 0) (param i32) (result i32)
(local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
global.get $__stack_pointer
i32.const 16
i32.sub
local.tee 1
global.set $__stack_pointer
block  ;; label = @1
    block  ;; label = @2
    block  ;; label = @3
        block  ;; label = @4
        block  ;; label = @5
            block  ;; label = @6
            block  ;; label = @7
                block  ;; label = @8
                block  ;; label = @9
                    block  ;; label = @10
                    block  ;; label = @11
                        block  ;; label = @12
                        i32.const 0
                        i32.load offset=82764
                        local.tee 2
                        br_if 0 (;@12;)
                        block  ;; label = @13
                            i32.const 0
                            i32.load offset=83212
                            local.tee 3
                            br_if 0 (;@13;)
                            i32.const 0
                            i64.const -1
                            i64.store offset=83224 align=4
                            i32.const 0
                            i64.const 281474976776192
                            i64.store offset=83216 align=4
                            i32.const 0
                            local.get 1
                            i32.const 8
                            i32.add
                            i32.const -16
                            i32.and
                            i32.const 1431655768
                            i32.xor
                            local.tee 3
                            i32.store offset=83212
                            i32.const 0
                            i32.const 0
                            i32.store offset=83232
                            i32.const 0
                            i32.const 0
                            i32.store offset=83184
                        end
                        i32.const 196608
                        i32.const 148784
                        i32.lt_u
                        br_if 1 (;@11;)
                        i32.const 0
                        local.set 2
                        i32.const 196608
                        i32.const 148784
                        i32.sub
                        i32.const 89
                        i32.lt_u
                        br_if 0 (;@12;)
                        i32.const 0
                        local.set 4
                        i32.const 0
                        i32.const 148784
                        i32.store offset=83188
                        i32.const 0
                        i32.const 148784
                        i32.store offset=82756
                        i32.const 0
                        local.get 3
                        i32.store offset=82776
                        i32.const 0
                        i32.const -1
                        i32.store offset=82772
                        i32.const 0
                        i32.const 196608
                        i32.const 148784
                        i32.sub
                        i32.store offset=83192
                        loop  ;; label = @13
                            local.get 4
                            i32.const 82800
                            i32.add
                            local.get 4
                            i32.const 82788
                            i32.add
                            local.tee 3
                            i32.store
                            local.get 3
                            local.get 4
                            i32.const 82780
                            i32.add
                            local.tee 5
                            i32.store
                            local.get 4
                            i32.const 82792
                            i32.add
                            local.get 5
                            i32.store
                            local.get 4
                            i32.const 82808
                            i32.add
                            local.get 4
                            i32.const 82796
                            i32.add
                            local.tee 5
                            i32.store
                            local.get 5
                            local.get 3
                            i32.store
                            local.get 4
                            i32.const 82816
                            i32.add
                            local.get 4
                            i32.const 82804
                            i32.add
                            local.tee 3
                            i32.store
                            local.get 3
                            local.get 5
                            i32.store
                            local.get 4
                            i32.const 82812
                            i32.add
                            local.get 3
                            i32.store
                            local.get 4
                            i32.const 32
                            i32.add
                            local.tee 4
                            i32.const 256
                            i32.ne
                            br_if 0 (;@13;)
                        end
                        i32.const 148784
                        i32.const -8
                        i32.const 148784
                        i32.sub
                        i32.const 15
                        i32.and
                        local.tee 4
                        i32.add
                        local.tee 2
                        i32.const 196608
                        i32.const 148784
                        i32.sub
                        i32.const -56
                        i32.add
                        local.tee 3
                        local.get 4
                        i32.sub
                        local.tee 4
                        i32.const 1
                        i32.or
                        i32.store offset=4
                        i32.const 0
                        i32.const 0
                        i32.load offset=83228
                        i32.store offset=82768
                        i32.const 0
                        local.get 4
                        i32.store offset=82752
                        i32.const 0
                        local.get 2
                        i32.store offset=82764
                        local.get 3
                        i32.const 148784
                        i32.add
                        i32.const 4
                        i32.add
                        i32.const 56
                        i32.store
                        end
                        block  ;; label = @12
                        block  ;; label = @13
                            local.get 0
                            i32.const 236
                            i32.gt_u
                            br_if 0 (;@13;)
                            block  ;; label = @14
                            i32.const 0
                            i32.load offset=82740
                            local.tee 6
                            i32.const 16
                            local.get 0
                            i32.const 19
                            i32.add
                            i32.const 496
                            i32.and
                            local.get 0
                            i32.const 11
                            i32.lt_u
                            select
                            local.tee 7
                            i32.const 3
                            i32.shr_u
                            local.tee 3
                            i32.shr_u
                            local.tee 4
                            i32.const 3
                            i32.and
                            i32.eqz
                            br_if 0 (;@14;)
                            block  ;; label = @15
                                block  ;; label = @16
                                local.get 4
                                i32.const 1
                                i32.and
                                local.get 3
                                i32.or
                                i32.const 1
                                i32.xor
                                local.tee 5
                                i32.const 3
                                i32.shl
                                local.tee 3
                                i32.const 82780
                                i32.add
                                local.tee 4
                                local.get 3
                                i32.const 82788
                                i32.add
                                i32.load
                                local.tee 3
                                i32.load offset=8
                                local.tee 7
                                i32.ne
                                br_if 0 (;@16;)
                                i32.const 0
                                local.get 6
                                i32.const -2
                                local.get 5
                                i32.rotl
                                i32.and
                                i32.store offset=82740
                                br 1 (;@15;)
                                end
                                local.get 4
                                local.get 7
                                i32.store offset=8
                                local.get 7
                                local.get 4
                                i32.store offset=12
                            end
                            local.get 3
                            i32.const 8
                            i32.add
                            local.set 4
                            local.get 3
                            local.get 5
                            i32.const 3
                            i32.shl
                            local.tee 5
                            i32.const 3
                            i32.or
                            i32.store offset=4
                            local.get 3
                            local.get 5
                            i32.add
                            local.tee 3
                            local.get 3
                            i32.load offset=4
                            i32.const 1
                            i32.or
                            i32.store offset=4
                            br 13 (;@1;)
                            end
                            local.get 7
                            i32.const 0
                            i32.load offset=82748
                            local.tee 8
                            i32.le_u
                            br_if 1 (;@12;)
                            block  ;; label = @14
                            local.get 4
                            i32.eqz
                            br_if 0 (;@14;)
                            block  ;; label = @15
                                block  ;; label = @16
                                local.get 4
                                local.get 3
                                i32.shl
                                i32.const 2
                                local.get 3
                                i32.shl
                                local.tee 4
                                i32.const 0
                                local.get 4
                                i32.sub
                                i32.or
                                i32.and
                                i32.ctz
                                local.tee 3
                                i32.const 3
                                i32.shl
                                local.tee 4
                                i32.const 82780
                                i32.add
                                local.tee 5
                                local.get 4
                                i32.const 82788
                                i32.add
                                i32.load
                                local.tee 4
                                i32.load offset=8
                                local.tee 0
                                i32.ne
                                br_if 0 (;@16;)
                                i32.const 0
                                local.get 6
                                i32.const -2
                                local.get 3
                                i32.rotl
                                i32.and
                                local.tee 6
                                i32.store offset=82740
                                br 1 (;@15;)
                                end
                                local.get 5
                                local.get 0
                                i32.store offset=8
                                local.get 0
                                local.get 5
                                i32.store offset=12
                            end
                            local.get 4
                            local.get 7
                            i32.const 3
                            i32.or
                            i32.store offset=4
                            local.get 4
                            local.get 3
                            i32.const 3
                            i32.shl
                            local.tee 3
                            i32.add
                            local.get 3
                            local.get 7
                            i32.sub
                            local.tee 5
                            i32.store
                            local.get 4
                            local.get 7
                            i32.add
                            local.tee 0
                            local.get 5
                            i32.const 1
                            i32.or
                            i32.store offset=4
                            block  ;; label = @15
                                local.get 8
                                i32.eqz
                                br_if 0 (;@15;)
                                local.get 8
                                i32.const -8
                                i32.and
                                i32.const 82780
                                i32.add
                                local.set 7
                                i32.const 0
                                i32.load offset=82760
                                local.set 3
                                block  ;; label = @16
                                block  ;; label = @17
                                    local.get 6
                                    i32.const 1
                                    local.get 8
                                    i32.const 3
                                    i32.shr_u
                                    i32.shl
                                    local.tee 9
                                    i32.and
                                    br_if 0 (;@17;)
                                    i32.const 0
                                    local.get 6
                                    local.get 9
                                    i32.or
                                    i32.store offset=82740
                                    local.get 7
                                    local.set 9
                                    br 1 (;@16;)
                                end
                                local.get 7
                                i32.load offset=8
                                local.set 9
                                end
                                local.get 9
                                local.get 3
                                i32.store offset=12
                                local.get 7
                                local.get 3
                                i32.store offset=8
                                local.get 3
                                local.get 7
                                i32.store offset=12
                                local.get 3
                                local.get 9
                                i32.store offset=8
                            end
                            local.get 4
                            i32.const 8
                            i32.add
                            local.set 4
                            i32.const 0
                            local.get 0
                            i32.store offset=82760
                            i32.const 0
                            local.get 5
                            i32.store offset=82748
                            br 13 (;@1;)
                            end
                            i32.const 0
                            i32.load offset=82744
                            local.tee 10
                            i32.eqz
                            br_if 1 (;@12;)
                            local.get 10
                            i32.ctz
                            i32.const 2
                            i32.shl
                            i32.const 83044
                            i32.add
                            i32.load
                            local.tee 0
                            i32.load offset=4
                            i32.const -8
                            i32.and
                            local.get 7
                            i32.sub
                            local.set 3
                            local.get 0
                            local.set 5
                            block  ;; label = @14
                            loop  ;; label = @15
                                block  ;; label = @16
                                local.get 5
                                i32.load offset=16
                                local.tee 4
                                br_if 0 (;@16;)
                                local.get 5
                                i32.const 20
                                i32.add
                                i32.load
                                local.tee 4
                                i32.eqz
                                br_if 2 (;@14;)
                                end
                                local.get 4
                                i32.load offset=4
                                i32.const -8
                                i32.and
                                local.get 7
                                i32.sub
                                local.tee 5
                                local.get 3
                                local.get 5
                                local.get 3
                                i32.lt_u
                                local.tee 5
                                select
                                local.set 3
                                local.get 4
                                local.get 0
                                local.get 5
                                select
                                local.set 0
                                local.get 4
                                local.set 5
                                br 0 (;@15;)
                            end
                            end
                            local.get 0
                            i32.load offset=24
                            local.set 11
                            block  ;; label = @14
                            local.get 0
                            i32.load offset=12
                            local.tee 9
                            local.get 0
                            i32.eq
                            br_if 0 (;@14;)
                            local.get 0
                            i32.load offset=8
                            local.tee 4
                            i32.const 0
                            i32.load offset=82756
                            i32.lt_u
                            drop
                            local.get 9
                            local.get 4
                            i32.store offset=8
                            local.get 4
                            local.get 9
                            i32.store offset=12
                            br 12 (;@2;)
                            end
                            block  ;; label = @14
                            local.get 0
                            i32.const 20
                            i32.add
                            local.tee 5
                            i32.load
                            local.tee 4
                            br_if 0 (;@14;)
                            local.get 0
                            i32.load offset=16
                            local.tee 4
                            i32.eqz
                            br_if 4 (;@10;)
                            local.get 0
                            i32.const 16
                            i32.add
                            local.set 5
                            end
                            loop  ;; label = @14
                            local.get 5
                            local.set 2
                            local.get 4
                            local.tee 9
                            i32.const 20
                            i32.add
                            local.tee 5
                            i32.load
                            local.tee 4
                            br_if 0 (;@14;)
                            local.get 9
                            i32.const 16
                            i32.add
                            local.set 5
                            local.get 9
                            i32.load offset=16
                            local.tee 4
                            br_if 0 (;@14;)
                            end
                            local.get 2
                            i32.const 0
                            i32.store
                            br 11 (;@2;)
                        end
                        i32.const -1
                        local.set 7
                        local.get 0
                        i32.const -65
                        i32.gt_u
                        br_if 0 (;@12;)
                        local.get 0
                        i32.const 19
                        i32.add
                        local.tee 4
                        i32.const -16
                        i32.and
                        local.set 7
                        i32.const 0
                        i32.load offset=82744
                        local.tee 11
                        i32.eqz
                        br_if 0 (;@12;)
                        i32.const 0
                        local.set 8
                        block  ;; label = @13
                            local.get 7
                            i32.const 256
                            i32.lt_u
                            br_if 0 (;@13;)
                            i32.const 31
                            local.set 8
                            local.get 7
                            i32.const 16777215
                            i32.gt_u
                            br_if 0 (;@13;)
                            local.get 7
                            i32.const 38
                            local.get 4
                            i32.const 8
                            i32.shr_u
                            i32.clz
                            local.tee 4
                            i32.sub
                            i32.shr_u
                            i32.const 1
                            i32.and
                            local.get 4
                            i32.const 1
                            i32.shl
                            i32.sub
                            i32.const 62
                            i32.add
                            local.set 8
                        end
                        i32.const 0
                        local.get 7
                        i32.sub
                        local.set 3
                        block  ;; label = @13
                            block  ;; label = @14
                            block  ;; label = @15
                                block  ;; label = @16
                                local.get 8
                                i32.const 2
                                i32.shl
                                i32.const 83044
                                i32.add
                                i32.load
                                local.tee 5
                                br_if 0 (;@16;)
                                i32.const 0
                                local.set 4
                                i32.const 0
                                local.set 9
                                br 1 (;@15;)
                                end
                                i32.const 0
                                local.set 4
                                local.get 7
                                i32.const 0
                                i32.const 25
                                local.get 8
                                i32.const 1
                                i32.shr_u
                                i32.sub
                                local.get 8
                                i32.const 31
                                i32.eq
                                select
                                i32.shl
                                local.set 0
                                i32.const 0
                                local.set 9
                                loop  ;; label = @16
                                block  ;; label = @17
                                    local.get 5
                                    i32.load offset=4
                                    i32.const -8
                                    i32.and
                                    local.get 7
                                    i32.sub
                                    local.tee 6
                                    local.get 3
                                    i32.ge_u
                                    br_if 0 (;@17;)
                                    local.get 6
                                    local.set 3
                                    local.get 5
                                    local.set 9
                                    local.get 6
                                    br_if 0 (;@17;)
                                    i32.const 0
                                    local.set 3
                                    local.get 5
                                    local.set 9
                                    local.get 5
                                    local.set 4
                                    br 3 (;@14;)
                                end
                                local.get 4
                                local.get 5
                                i32.const 20
                                i32.add
                                i32.load
                                local.tee 6
                                local.get 6
                                local.get 5
                                local.get 0
                                i32.const 29
                                i32.shr_u
                                i32.const 4
                                i32.and
                                i32.add
                                i32.const 16
                                i32.add
                                i32.load
                                local.tee 5
                                i32.eq
                                select
                                local.get 4
                                local.get 6
                                select
                                local.set 4
                                local.get 0
                                i32.const 1
                                i32.shl
                                local.set 0
                                local.get 5
                                br_if 0 (;@16;)
                                end
                            end
                            block  ;; label = @15
                                local.get 4
                                local.get 9
                                i32.or
                                br_if 0 (;@15;)
                                i32.const 0
                                local.set 9
                                i32.const 2
                                local.get 8
                                i32.shl
                                local.tee 4
                                i32.const 0
                                local.get 4
                                i32.sub
                                i32.or
                                local.get 11
                                i32.and
                                local.tee 4
                                i32.eqz
                                br_if 3 (;@12;)
                                local.get 4
                                i32.ctz
                                i32.const 2
                                i32.shl
                                i32.const 83044
                                i32.add
                                i32.load
                                local.set 4
                            end
                            local.get 4
                            i32.eqz
                            br_if 1 (;@13;)
                            end
                            loop  ;; label = @14
                            local.get 4
                            i32.load offset=4
                            i32.const -8
                            i32.and
                            local.get 7
                            i32.sub
                            local.tee 6
                            local.get 3
                            i32.lt_u
                            local.set 0
                            block  ;; label = @15
                                local.get 4
                                i32.load offset=16
                                local.tee 5
                                br_if 0 (;@15;)
                                local.get 4
                                i32.const 20
                                i32.add
                                i32.load
                                local.set 5
                            end
                            local.get 6
                            local.get 3
                            local.get 0
                            select
                            local.set 3
                            local.get 4
                            local.get 9
                            local.get 0
                            select
                            local.set 9
                            local.get 5
                            local.set 4
                            local.get 5
                            br_if 0 (;@14;)
                            end
                        end
                        local.get 9
                        i32.eqz
                        br_if 0 (;@12;)
                        local.get 3
                        i32.const 0
                        i32.load offset=82748
                        local.get 7
                        i32.sub
                        i32.ge_u
                        br_if 0 (;@12;)
                        local.get 9
                        i32.load offset=24
                        local.set 2
                        block  ;; label = @13
                            local.get 9
                            i32.load offset=12
                            local.tee 0
                            local.get 9
                            i32.eq
                            br_if 0 (;@13;)
                            local.get 9
                            i32.load offset=8
                            local.tee 4
                            i32.const 0
                            i32.load offset=82756
                            i32.lt_u
                            drop
                            local.get 0
                            local.get 4
                            i32.store offset=8
                            local.get 4
                            local.get 0
                            i32.store offset=12
                            br 10 (;@3;)
                        end
                        block  ;; label = @13
                            local.get 9
                            i32.const 20
                            i32.add
                            local.tee 5
                            i32.load
                            local.tee 4
                            br_if 0 (;@13;)
                            local.get 9
                            i32.load offset=16
                            local.tee 4
                            i32.eqz
                            br_if 4 (;@9;)
                            local.get 9
                            i32.const 16
                            i32.add
                            local.set 5
                        end
                        loop  ;; label = @13
                            local.get 5
                            local.set 6
                            local.get 4
                            local.tee 0
                            i32.const 20
                            i32.add
                            local.tee 5
                            i32.load
                            local.tee 4
                            br_if 0 (;@13;)
                            local.get 0
                            i32.const 16
                            i32.add
                            local.set 5
                            local.get 0
                            i32.load offset=16
                            local.tee 4
                            br_if 0 (;@13;)
                        end
                        local.get 6
                        i32.const 0
                        i32.store
                        br 9 (;@3;)
                        end
                        block  ;; label = @12
                        i32.const 0
                        i32.load offset=82748
                        local.tee 4
                        local.get 7
                        i32.lt_u
                        br_if 0 (;@12;)
                        i32.const 0
                        i32.load offset=82760
                        local.set 3
                        block  ;; label = @13
                            block  ;; label = @14
                            local.get 4
                            local.get 7
                            i32.sub
                            local.tee 5
                            i32.const 16
                            i32.lt_u
                            br_if 0 (;@14;)
                            local.get 3
                            local.get 7
                            i32.add
                            local.tee 0
                            local.get 5
                            i32.const 1
                            i32.or
                            i32.store offset=4
                            local.get 3
                            local.get 4
                            i32.add
                            local.get 5
                            i32.store
                            local.get 3
                            local.get 7
                            i32.const 3
                            i32.or
                            i32.store offset=4
                            br 1 (;@13;)
                            end
                            local.get 3
                            local.get 4
                            i32.const 3
                            i32.or
                            i32.store offset=4
                            local.get 3
                            local.get 4
                            i32.add
                            local.tee 4
                            local.get 4
                            i32.load offset=4
                            i32.const 1
                            i32.or
                            i32.store offset=4
                            i32.const 0
                            local.set 0
                            i32.const 0
                            local.set 5
                        end
                        i32.const 0
                        local.get 5
                        i32.store offset=82748
                        i32.const 0
                        local.get 0
                        i32.store offset=82760
                        local.get 3
                        i32.const 8
                        i32.add
                        local.set 4
                        br 11 (;@1;)
                        end
                        block  ;; label = @12
                        i32.const 0
                        i32.load offset=82752
                        local.tee 5
                        local.get 7
                        i32.le_u
                        br_if 0 (;@12;)
                        local.get 2
                        local.get 7
                        i32.add
                        local.tee 4
                        local.get 5
                        local.get 7
                        i32.sub
                        local.tee 3
                        i32.const 1
                        i32.or
                        i32.store offset=4
                        i32.const 0
                        local.get 4
                        i32.store offset=82764
                        i32.const 0
                        local.get 3
                        i32.store offset=82752
                        local.get 2
                        local.get 7
                        i32.const 3
                        i32.or
                        i32.store offset=4
                        local.get 2
                        i32.const 8
                        i32.add
                        local.set 4
                        br 11 (;@1;)
                        end
                        block  ;; label = @12
                        block  ;; label = @13
                            i32.const 0
                            i32.load offset=83212
                            i32.eqz
                            br_if 0 (;@13;)
                            i32.const 0
                            i32.load offset=83220
                            local.set 3
                            br 1 (;@12;)
                        end
                        i32.const 0
                        i64.const -1
                        i64.store offset=83224 align=4
                        i32.const 0
                        i64.const 281474976776192
                        i64.store offset=83216 align=4
                        i32.const 0
                        local.get 1
                        i32.const 12
                        i32.add
                        i32.const -16
                        i32.and
                        i32.const 1431655768
                        i32.xor
                        i32.store offset=83212
                        i32.const 0
                        i32.const 0
                        i32.store offset=83232
                        i32.const 0
                        i32.const 0
                        i32.store offset=83184
                        i32.const 65536
                        local.set 3
                        end
                        i32.const 0
                        local.set 4
                        block  ;; label = @12
                        local.get 3
                        local.get 7
                        i32.const 71
                        i32.add
                        local.tee 8
                        i32.add
                        local.tee 0
                        i32.const 0
                        local.get 3
                        i32.sub
                        local.tee 6
                        i32.and
                        local.tee 9
                        local.get 7
                        i32.gt_u
                        br_if 0 (;@12;)
                        i32.const 0
                        i32.const 48
                        i32.store offset=82736
                        br 11 (;@1;)
                        end
                        block  ;; label = @12
                        i32.const 0
                        i32.load offset=83180
                        local.tee 4
                        i32.eqz
                        br_if 0 (;@12;)
                        block  ;; label = @13
                            i32.const 0
                            i32.load offset=83172
                            local.tee 3
                            local.get 9
                            i32.add
                            local.tee 11
                            local.get 3
                            i32.le_u
                            br_if 0 (;@13;)
                            local.get 11
                            local.get 4
                            i32.le_u
                            br_if 1 (;@12;)
                        end
                        i32.const 0
                        local.set 4
                        i32.const 0
                        i32.const 48
                        i32.store offset=82736
                        br 11 (;@1;)
                        end
                        i32.const 0
                        i32.load8_u offset=83184
                        i32.const 4
                        i32.and
                        br_if 5 (;@6;)
                        block  ;; label = @12
                        block  ;; label = @13
                            block  ;; label = @14
                            local.get 2
                            i32.eqz
                            br_if 0 (;@14;)
                            i32.const 83188
                            local.set 4
                            loop  ;; label = @15
                                block  ;; label = @16
                                local.get 4
                                i32.load
                                local.tee 3
                                local.get 2
                                i32.gt_u
                                br_if 0 (;@16;)
                                local.get 3
                                local.get 4
                                i32.load offset=4
                                i32.add
                                local.get 2
                                i32.gt_u
                                br_if 3 (;@13;)
                                end
                                local.get 4
                                i32.load offset=8
                                local.tee 4
                                br_if 0 (;@15;)
                            end
                            end
                            i32.const 0
                            call $sbrk
                            local.tee 0
                            i32.const -1
                            i32.eq
                            br_if 6 (;@7;)
                            local.get 9
                            local.set 6
                            block  ;; label = @14
                            i32.const 0
                            i32.load offset=83216
                            local.tee 4
                            i32.const -1
                            i32.add
                            local.tee 3
                            local.get 0
                            i32.and
                            i32.eqz
                            br_if 0 (;@14;)
                            local.get 9
                            local.get 0
                            i32.sub
                            local.get 3
                            local.get 0
                            i32.add
                            i32.const 0
                            local.get 4
                            i32.sub
                            i32.and
                            i32.add
                            local.set 6
                            end
                            local.get 6
                            local.get 7
                            i32.le_u
                            br_if 6 (;@7;)
                            local.get 6
                            i32.const 2147483646
                            i32.gt_u
                            br_if 6 (;@7;)
                            block  ;; label = @14
                            i32.const 0
                            i32.load offset=83180
                            local.tee 4
                            i32.eqz
                            br_if 0 (;@14;)
                            i32.const 0
                            i32.load offset=83172
                            local.tee 3
                            local.get 6
                            i32.add
                            local.tee 5
                            local.get 3
                            i32.le_u
                            br_if 7 (;@7;)
                            local.get 5
                            local.get 4
                            i32.gt_u
                            br_if 7 (;@7;)
                            end
                            local.get 6
                            call $sbrk
                            local.tee 4
                            local.get 0
                            i32.ne
                            br_if 1 (;@12;)
                            br 8 (;@5;)
                        end
                        local.get 0
                        local.get 5
                        i32.sub
                        local.get 6
                        i32.and
                        local.tee 6
                        i32.const 2147483646
                        i32.gt_u
                        br_if 5 (;@7;)
                        local.get 6
                        call $sbrk
                        local.tee 0
                        local.get 4
                        i32.load
                        local.get 4
                        i32.load offset=4
                        i32.add
                        i32.eq
                        br_if 4 (;@8;)
                        local.get 0
                        local.set 4
                        end
                        block  ;; label = @12
                        local.get 6
                        local.get 7
                        i32.const 72
                        i32.add
                        i32.ge_u
                        br_if 0 (;@12;)
                        local.get 4
                        i32.const -1
                        i32.eq
                        br_if 0 (;@12;)
                        block  ;; label = @13
                            local.get 8
                            local.get 6
                            i32.sub
                            i32.const 0
                            i32.load offset=83220
                            local.tee 3
                            i32.add
                            i32.const 0
                            local.get 3
                            i32.sub
                            i32.and
                            local.tee 3
                            i32.const 2147483646
                            i32.le_u
                            br_if 0 (;@13;)
                            local.get 4
                            local.set 0
                            br 8 (;@5;)
                        end
                        block  ;; label = @13
                            local.get 3
                            call $sbrk
                            i32.const -1
                            i32.eq
                            br_if 0 (;@13;)
                            local.get 3
                            local.get 6
                            i32.add
                            local.set 6
                            local.get 4
                            local.set 0
                            br 8 (;@5;)
                        end
                        i32.const 0
                        local.get 6
                        i32.sub
                        call $sbrk
                        drop
                        br 5 (;@7;)
                        end
                        local.get 4
                        local.set 0
                        local.get 4
                        i32.const -1
                        i32.ne
                        br_if 6 (;@5;)
                        br 4 (;@7;)
                    end
                    unreachable
                    unreachable
                    end
                    i32.const 0
                    local.set 9
                    br 7 (;@2;)
                end
                i32.const 0
                local.set 0
                br 5 (;@3;)
                end
                local.get 0
                i32.const -1
                i32.ne
                br_if 2 (;@5;)
            end
            i32.const 0
            i32.const 0
            i32.load offset=83184
            i32.const 4
            i32.or
            i32.store offset=83184
            end
            local.get 9
            i32.const 2147483646
            i32.gt_u
            br_if 1 (;@4;)
            local.get 9
            call $sbrk
            local.set 0
            i32.const 0
            call $sbrk
            local.set 4
            local.get 0
            i32.const -1
            i32.eq
            br_if 1 (;@4;)
            local.get 4
            i32.const -1
            i32.eq
            br_if 1 (;@4;)
            local.get 0
            local.get 4
            i32.ge_u
            br_if 1 (;@4;)
            local.get 4
            local.get 0
            i32.sub
            local.tee 6
            local.get 7
            i32.const 56
            i32.add
            i32.le_u
            br_if 1 (;@4;)
        end
        i32.const 0
        i32.const 0
        i32.load offset=83172
        local.get 6
        i32.add
        local.tee 4
        i32.store offset=83172
        block  ;; label = @5
            local.get 4
            i32.const 0
            i32.load offset=83176
            i32.le_u
            br_if 0 (;@5;)
            i32.const 0
            local.get 4
            i32.store offset=83176
        end
        block  ;; label = @5
            block  ;; label = @6
            block  ;; label = @7
                block  ;; label = @8
                i32.const 0
                i32.load offset=82764
                local.tee 3
                i32.eqz
                br_if 0 (;@8;)
                i32.const 83188
                local.set 4
                loop  ;; label = @9
                    local.get 0
                    local.get 4
                    i32.load
                    local.tee 5
                    local.get 4
                    i32.load offset=4
                    local.tee 9
                    i32.add
                    i32.eq
                    br_if 2 (;@7;)
                    local.get 4
                    i32.load offset=8
                    local.tee 4
                    br_if 0 (;@9;)
                    br 3 (;@6;)
                end
                end
                block  ;; label = @8
                block  ;; label = @9
                    i32.const 0
                    i32.load offset=82756
                    local.tee 4
                    i32.eqz
                    br_if 0 (;@9;)
                    local.get 0
                    local.get 4
                    i32.ge_u
                    br_if 1 (;@8;)
                end
                i32.const 0
                local.get 0
                i32.store offset=82756
                end
                i32.const 0
                local.set 4
                i32.const 0
                local.get 6
                i32.store offset=83192
                i32.const 0
                local.get 0
                i32.store offset=83188
                i32.const 0
                i32.const -1
                i32.store offset=82772
                i32.const 0
                i32.const 0
                i32.load offset=83212
                i32.store offset=82776
                i32.const 0
                i32.const 0
                i32.store offset=83200
                loop  ;; label = @8
                local.get 4
                i32.const 82800
                i32.add
                local.get 4
                i32.const 82788
                i32.add
                local.tee 3
                i32.store
                local.get 3
                local.get 4
                i32.const 82780
                i32.add
                local.tee 5
                i32.store
                local.get 4
                i32.const 82792
                i32.add
                local.get 5
                i32.store
                local.get 4
                i32.const 82808
                i32.add
                local.get 4
                i32.const 82796
                i32.add
                local.tee 5
                i32.store
                local.get 5
                local.get 3
                i32.store
                local.get 4
                i32.const 82816
                i32.add
                local.get 4
                i32.const 82804
                i32.add
                local.tee 3
                i32.store
                local.get 3
                local.get 5
                i32.store
                local.get 4
                i32.const 82812
                i32.add
                local.get 3
                i32.store
                local.get 4
                i32.const 32
                i32.add
                local.tee 4
                i32.const 256
                i32.ne
                br_if 0 (;@8;)
                end
                local.get 0
                i32.const -8
                local.get 0
                i32.sub
                i32.const 15
                i32.and
                local.tee 4
                i32.add
                local.tee 3
                local.get 6
                i32.const -56
                i32.add
                local.tee 5
                local.get 4
                i32.sub
                local.tee 4
                i32.const 1
                i32.or
                i32.store offset=4
                i32.const 0
                i32.const 0
                i32.load offset=83228
                i32.store offset=82768
                i32.const 0
                local.get 4
                i32.store offset=82752
                i32.const 0
                local.get 3
                i32.store offset=82764
                local.get 0
                local.get 5
                i32.add
                i32.const 56
                i32.store offset=4
                br 2 (;@5;)
            end
            local.get 3
            local.get 0
            i32.ge_u
            br_if 0 (;@6;)
            local.get 3
            local.get 5
            i32.lt_u
            br_if 0 (;@6;)
            local.get 4
            i32.load offset=12
            i32.const 8
            i32.and
            br_if 0 (;@6;)
            local.get 3
            i32.const -8
            local.get 3
            i32.sub
            i32.const 15
            i32.and
            local.tee 5
            i32.add
            local.tee 0
            i32.const 0
            i32.load offset=82752
            local.get 6
            i32.add
            local.tee 2
            local.get 5
            i32.sub
            local.tee 5
            i32.const 1
            i32.or
            i32.store offset=4
            local.get 4
            local.get 9
            local.get 6
            i32.add
            i32.store offset=4
            i32.const 0
            i32.const 0
            i32.load offset=83228
            i32.store offset=82768
            i32.const 0
            local.get 5
            i32.store offset=82752
            i32.const 0
            local.get 0
            i32.store offset=82764
            local.get 3
            local.get 2
            i32.add
            i32.const 56
            i32.store offset=4
            br 1 (;@5;)
            end
            block  ;; label = @6
            local.get 0
            i32.const 0
            i32.load offset=82756
            i32.ge_u
            br_if 0 (;@6;)
            i32.const 0
            local.get 0
            i32.store offset=82756
            end
            local.get 0
            local.get 6
            i32.add
            local.set 5
            i32.const 83188
            local.set 4
            block  ;; label = @6
            block  ;; label = @7
                block  ;; label = @8
                block  ;; label = @9
                    loop  ;; label = @10
                    local.get 4
                    i32.load
                    local.get 5
                    i32.eq
                    br_if 1 (;@9;)
                    local.get 4
                    i32.load offset=8
                    local.tee 4
                    br_if 0 (;@10;)
                    br 2 (;@8;)
                    end
                end
                local.get 4
                i32.load8_u offset=12
                i32.const 8
                i32.and
                i32.eqz
                br_if 1 (;@7;)
                end
                i32.const 83188
                local.set 4
                block  ;; label = @8
                loop  ;; label = @9
                    block  ;; label = @10
                    local.get 4
                    i32.load
                    local.tee 5
                    local.get 3
                    i32.gt_u
                    br_if 0 (;@10;)
                    local.get 5
                    local.get 4
                    i32.load offset=4
                    i32.add
                    local.tee 5
                    local.get 3
                    i32.gt_u
                    br_if 2 (;@8;)
                    end
                    local.get 4
                    i32.load offset=8
                    local.set 4
                    br 0 (;@9;)
                end
                end
                local.get 0
                i32.const -8
                local.get 0
                i32.sub
                i32.const 15
                i32.and
                local.tee 4
                i32.add
                local.tee 2
                local.get 6
                i32.const -56
                i32.add
                local.tee 9
                local.get 4
                i32.sub
                local.tee 4
                i32.const 1
                i32.or
                i32.store offset=4
                local.get 0
                local.get 9
                i32.add
                i32.const 56
                i32.store offset=4
                local.get 3
                local.get 5
                i32.const 55
                local.get 5
                i32.sub
                i32.const 15
                i32.and
                i32.add
                i32.const -63
                i32.add
                local.tee 9
                local.get 9
                local.get 3
                i32.const 16
                i32.add
                i32.lt_u
                select
                local.tee 9
                i32.const 35
                i32.store offset=4
                i32.const 0
                i32.const 0
                i32.load offset=83228
                i32.store offset=82768
                i32.const 0
                local.get 4
                i32.store offset=82752
                i32.const 0
                local.get 2
                i32.store offset=82764
                local.get 9
                i32.const 16
                i32.add
                i32.const 0
                i64.load offset=83196 align=4
                i64.store align=4
                local.get 9
                i32.const 0
                i64.load offset=83188 align=4
                i64.store offset=8 align=4
                i32.const 0
                local.get 9
                i32.const 8
                i32.add
                i32.store offset=83196
                i32.const 0
                local.get 6
                i32.store offset=83192
                i32.const 0
                local.get 0
                i32.store offset=83188
                i32.const 0
                i32.const 0
                i32.store offset=83200
                local.get 9
                i32.const 36
                i32.add
                local.set 4
                loop  ;; label = @8
                local.get 4
                i32.const 7
                i32.store
                local.get 4
                i32.const 4
                i32.add
                local.tee 4
                local.get 5
                i32.lt_u
                br_if 0 (;@8;)
                end
                local.get 9
                local.get 3
                i32.eq
                br_if 2 (;@5;)
                local.get 9
                local.get 9
                i32.load offset=4
                i32.const -2
                i32.and
                i32.store offset=4
                local.get 9
                local.get 9
                local.get 3
                i32.sub
                local.tee 0
                i32.store
                local.get 3
                local.get 0
                i32.const 1
                i32.or
                i32.store offset=4
                block  ;; label = @8
                local.get 0
                i32.const 255
                i32.gt_u
                br_if 0 (;@8;)
                local.get 0
                i32.const -8
                i32.and
                i32.const 82780
                i32.add
                local.set 4
                block  ;; label = @9
                    block  ;; label = @10
                    i32.const 0
                    i32.load offset=82740
                    local.tee 5
                    i32.const 1
                    local.get 0
                    i32.const 3
                    i32.shr_u
                    i32.shl
                    local.tee 0
                    i32.and
                    br_if 0 (;@10;)
                    i32.const 0
                    local.get 5
                    local.get 0
                    i32.or
                    i32.store offset=82740
                    local.get 4
                    local.set 5
                    br 1 (;@9;)
                    end
                    local.get 4
                    i32.load offset=8
                    local.set 5
                end
                local.get 5
                local.get 3
                i32.store offset=12
                local.get 4
                local.get 3
                i32.store offset=8
                local.get 3
                local.get 4
                i32.store offset=12
                local.get 3
                local.get 5
                i32.store offset=8
                br 3 (;@5;)
                end
                i32.const 31
                local.set 4
                block  ;; label = @8
                local.get 0
                i32.const 16777215
                i32.gt_u
                br_if 0 (;@8;)
                local.get 0
                i32.const 38
                local.get 0
                i32.const 8
                i32.shr_u
                i32.clz
                local.tee 4
                i32.sub
                i32.shr_u
                i32.const 1
                i32.and
                local.get 4
                i32.const 1
                i32.shl
                i32.sub
                i32.const 62
                i32.add
                local.set 4
                end
                local.get 3
                local.get 4
                i32.store offset=28
                local.get 3
                i64.const 0
                i64.store offset=16 align=4
                local.get 4
                i32.const 2
                i32.shl
                i32.const 83044
                i32.add
                local.set 5
                block  ;; label = @8
                i32.const 0
                i32.load offset=82744
                local.tee 9
                i32.const 1
                local.get 4
                i32.shl
                local.tee 6
                i32.and
                br_if 0 (;@8;)
                local.get 5
                local.get 3
                i32.store
                i32.const 0
                local.get 9
                local.get 6
                i32.or
                i32.store offset=82744
                local.get 3
                local.get 5
                i32.store offset=24
                local.get 3
                local.get 3
                i32.store offset=8
                local.get 3
                local.get 3
                i32.store offset=12
                br 3 (;@5;)
                end
                local.get 0
                i32.const 0
                i32.const 25
                local.get 4
                i32.const 1
                i32.shr_u
                i32.sub
                local.get 4
                i32.const 31
                i32.eq
                select
                i32.shl
                local.set 4
                local.get 5
                i32.load
                local.set 9
                loop  ;; label = @8
                local.get 9
                local.tee 5
                i32.load offset=4
                i32.const -8
                i32.and
                local.get 0
                i32.eq
                br_if 2 (;@6;)
                local.get 4
                i32.const 29
                i32.shr_u
                local.set 9
                local.get 4
                i32.const 1
                i32.shl
                local.set 4
                local.get 5
                local.get 9
                i32.const 4
                i32.and
                i32.add
                i32.const 16
                i32.add
                local.tee 6
                i32.load
                local.tee 9
                br_if 0 (;@8;)
                end
                local.get 6
                local.get 3
                i32.store
                local.get 3
                local.get 5
                i32.store offset=24
                local.get 3
                local.get 3
                i32.store offset=12
                local.get 3
                local.get 3
                i32.store offset=8
                br 2 (;@5;)
            end
            local.get 4
            local.get 0
            i32.store
            local.get 4
            local.get 4
            i32.load offset=4
            local.get 6
            i32.add
            i32.store offset=4
            local.get 0
            local.get 5
            local.get 7
            call $prepend_alloc
            local.set 4
            br 5 (;@1;)
            end
            local.get 5
            i32.load offset=8
            local.tee 4
            local.get 3
            i32.store offset=12
            local.get 5
            local.get 3
            i32.store offset=8
            local.get 3
            i32.const 0
            i32.store offset=24
            local.get 3
            local.get 5
            i32.store offset=12
            local.get 3
            local.get 4
            i32.store offset=8
        end
        i32.const 0
        i32.load offset=82752
        local.tee 4
        local.get 7
        i32.le_u
        br_if 0 (;@4;)
        i32.const 0
        i32.load offset=82764
        local.tee 3
        local.get 7
        i32.add
        local.tee 5
        local.get 4
        local.get 7
        i32.sub
        local.tee 4
        i32.const 1
        i32.or
        i32.store offset=4
        i32.const 0
        local.get 4
        i32.store offset=82752
        i32.const 0
        local.get 5
        i32.store offset=82764
        local.get 3
        local.get 7
        i32.const 3
        i32.or
        i32.store offset=4
        local.get 3
        i32.const 8
        i32.add
        local.set 4
        br 3 (;@1;)
        end
        i32.const 0
        local.set 4
        i32.const 0
        i32.const 48
        i32.store offset=82736
        br 2 (;@1;)
    end
    block  ;; label = @3
        local.get 2
        i32.eqz
        br_if 0 (;@3;)
        block  ;; label = @4
        block  ;; label = @5
            local.get 9
            local.get 9
            i32.load offset=28
            local.tee 5
            i32.const 2
            i32.shl
            i32.const 83044
            i32.add
            local.tee 4
            i32.load
            i32.ne
            br_if 0 (;@5;)
            local.get 4
            local.get 0
            i32.store
            local.get 0
            br_if 1 (;@4;)
            i32.const 0
            local.get 11
            i32.const -2
            local.get 5
            i32.rotl
            i32.and
            local.tee 11
            i32.store offset=82744
            br 2 (;@3;)
        end
        local.get 2
        i32.const 16
        i32.const 20
        local.get 2
        i32.load offset=16
        local.get 9
        i32.eq
        select
        i32.add
        local.get 0
        i32.store
        local.get 0
        i32.eqz
        br_if 1 (;@3;)
        end
        local.get 0
        local.get 2
        i32.store offset=24
        block  ;; label = @4
        local.get 9
        i32.load offset=16
        local.tee 4
        i32.eqz
        br_if 0 (;@4;)
        local.get 0
        local.get 4
        i32.store offset=16
        local.get 4
        local.get 0
        i32.store offset=24
        end
        local.get 9
        i32.const 20
        i32.add
        i32.load
        local.tee 4
        i32.eqz
        br_if 0 (;@3;)
        local.get 0
        i32.const 20
        i32.add
        local.get 4
        i32.store
        local.get 4
        local.get 0
        i32.store offset=24
    end
    block  ;; label = @3
        block  ;; label = @4
        local.get 3
        i32.const 15
        i32.gt_u
        br_if 0 (;@4;)
        local.get 9
        local.get 3
        local.get 7
        i32.or
        local.tee 4
        i32.const 3
        i32.or
        i32.store offset=4
        local.get 9
        local.get 4
        i32.add
        local.tee 4
        local.get 4
        i32.load offset=4
        i32.const 1
        i32.or
        i32.store offset=4
        br 1 (;@3;)
        end
        local.get 9
        local.get 7
        i32.add
        local.tee 0
        local.get 3
        i32.const 1
        i32.or
        i32.store offset=4
        local.get 9
        local.get 7
        i32.const 3
        i32.or
        i32.store offset=4
        local.get 0
        local.get 3
        i32.add
        local.get 3
        i32.store
        block  ;; label = @4
        local.get 3
        i32.const 255
        i32.gt_u
        br_if 0 (;@4;)
        local.get 3
        i32.const -8
        i32.and
        i32.const 82780
        i32.add
        local.set 4
        block  ;; label = @5
            block  ;; label = @6
            i32.const 0
            i32.load offset=82740
            local.tee 5
            i32.const 1
            local.get 3
            i32.const 3
            i32.shr_u
            i32.shl
            local.tee 3
            i32.and
            br_if 0 (;@6;)
            i32.const 0
            local.get 5
            local.get 3
            i32.or
            i32.store offset=82740
            local.get 4
            local.set 3
            br 1 (;@5;)
            end
            local.get 4
            i32.load offset=8
            local.set 3
        end
        local.get 3
        local.get 0
        i32.store offset=12
        local.get 4
        local.get 0
        i32.store offset=8
        local.get 0
        local.get 4
        i32.store offset=12
        local.get 0
        local.get 3
        i32.store offset=8
        br 1 (;@3;)
        end
        i32.const 31
        local.set 4
        block  ;; label = @4
        local.get 3
        i32.const 16777215
        i32.gt_u
        br_if 0 (;@4;)
        local.get 3
        i32.const 38
        local.get 3
        i32.const 8
        i32.shr_u
        i32.clz
        local.tee 4
        i32.sub
        i32.shr_u
        i32.const 1
        i32.and
        local.get 4
        i32.const 1
        i32.shl
        i32.sub
        i32.const 62
        i32.add
        local.set 4
        end
        local.get 0
        local.get 4
        i32.store offset=28
        local.get 0
        i64.const 0
        i64.store offset=16 align=4
        local.get 4
        i32.const 2
        i32.shl
        i32.const 83044
        i32.add
        local.set 5
        block  ;; label = @4
        local.get 11
        i32.const 1
        local.get 4
        i32.shl
        local.tee 7
        i32.and
        br_if 0 (;@4;)
        local.get 5
        local.get 0
        i32.store
        i32.const 0
        local.get 11
        local.get 7
        i32.or
        i32.store offset=82744
        local.get 0
        local.get 5
        i32.store offset=24
        local.get 0
        local.get 0
        i32.store offset=8
        local.get 0
        local.get 0
        i32.store offset=12
        br 1 (;@3;)
        end
        local.get 3
        i32.const 0
        i32.const 25
        local.get 4
        i32.const 1
        i32.shr_u
        i32.sub
        local.get 4
        i32.const 31
        i32.eq
        select
        i32.shl
        local.set 4
        local.get 5
        i32.load
        local.set 7
        block  ;; label = @4
        loop  ;; label = @5
            local.get 7
            local.tee 5
            i32.load offset=4
            i32.const -8
            i32.and
            local.get 3
            i32.eq
            br_if 1 (;@4;)
            local.get 4
            i32.const 29
            i32.shr_u
            local.set 7
            local.get 4
            i32.const 1
            i32.shl
            local.set 4
            local.get 5
            local.get 7
            i32.const 4
            i32.and
            i32.add
            i32.const 16
            i32.add
            local.tee 6
            i32.load
            local.tee 7
            br_if 0 (;@5;)
        end
        local.get 6
        local.get 0
        i32.store
        local.get 0
        local.get 5
        i32.store offset=24
        local.get 0
        local.get 0
        i32.store offset=12
        local.get 0
        local.get 0
        i32.store offset=8
        br 1 (;@3;)
        end
        local.get 5
        i32.load offset=8
        local.tee 4
        local.get 0
        i32.store offset=12
        local.get 5
        local.get 0
        i32.store offset=8
        local.get 0
        i32.const 0
        i32.store offset=24
        local.get 0
        local.get 5
        i32.store offset=12
        local.get 0
        local.get 4
        i32.store offset=8
    end
    local.get 9
    i32.const 8
    i32.add
    local.set 4
    br 1 (;@1;)
    end
    block  ;; label = @2
    local.get 11
    i32.eqz
    br_if 0 (;@2;)
    block  ;; label = @3
        block  ;; label = @4
        local.get 0
        local.get 0
        i32.load offset=28
        local.tee 5
        i32.const 2
        i32.shl
        i32.const 83044
        i32.add
        local.tee 4
        i32.load
        i32.ne
        br_if 0 (;@4;)
        local.get 4
        local.get 9
        i32.store
        local.get 9
        br_if 1 (;@3;)
        i32.const 0
        local.get 10
        i32.const -2
        local.get 5
        i32.rotl
        i32.and
        i32.store offset=82744
        br 2 (;@2;)
        end
        local.get 11
        i32.const 16
        i32.const 20
        local.get 11
        i32.load offset=16
        local.get 0
        i32.eq
        select
        i32.add
        local.get 9
        i32.store
        local.get 9
        i32.eqz
        br_if 1 (;@2;)
    end
    local.get 9
    local.get 11
    i32.store offset=24
    block  ;; label = @3
        local.get 0
        i32.load offset=16
        local.tee 4
        i32.eqz
        br_if 0 (;@3;)
        local.get 9
        local.get 4
        i32.store offset=16
        local.get 4
        local.get 9
        i32.store offset=24
    end
    local.get 0
    i32.const 20
    i32.add
    i32.load
    local.tee 4
    i32.eqz
    br_if 0 (;@2;)
    local.get 9
    i32.const 20
    i32.add
    local.get 4
    i32.store
    local.get 4
    local.get 9
    i32.store offset=24
    end
    block  ;; label = @2
    block  ;; label = @3
        local.get 3
        i32.const 15
        i32.gt_u
        br_if 0 (;@3;)
        local.get 0
        local.get 3
        local.get 7
        i32.or
        local.tee 4
        i32.const 3
        i32.or
        i32.store offset=4
        local.get 0
        local.get 4
        i32.add
        local.tee 4
        local.get 4
        i32.load offset=4
        i32.const 1
        i32.or
        i32.store offset=4
        br 1 (;@2;)
    end
    local.get 0
    local.get 7
    i32.add
    local.tee 5
    local.get 3
    i32.const 1
    i32.or
    i32.store offset=4
    local.get 0
    local.get 7
    i32.const 3
    i32.or
    i32.store offset=4
    local.get 5
    local.get 3
    i32.add
    local.get 3
    i32.store
    block  ;; label = @3
        local.get 8
        i32.eqz
        br_if 0 (;@3;)
        local.get 8
        i32.const -8
        i32.and
        i32.const 82780
        i32.add
        local.set 7
        i32.const 0
        i32.load offset=82760
        local.set 4
        block  ;; label = @4
        block  ;; label = @5
            i32.const 1
            local.get 8
            i32.const 3
            i32.shr_u
            i32.shl
            local.tee 9
            local.get 6
            i32.and
            br_if 0 (;@5;)
            i32.const 0
            local.get 9
            local.get 6
            i32.or
            i32.store offset=82740
            local.get 7
            local.set 9
            br 1 (;@4;)
        end
        local.get 7
        i32.load offset=8
        local.set 9
        end
        local.get 9
        local.get 4
        i32.store offset=12
        local.get 7
        local.get 4
        i32.store offset=8
        local.get 4
        local.get 7
        i32.store offset=12
        local.get 4
        local.get 9
        i32.store offset=8
    end
    i32.const 0
    local.get 5
    i32.store offset=82760
    i32.const 0
    local.get 3
    i32.store offset=82748
    end
    local.get 0
    i32.const 8
    i32.add
    local.set 4
end
local.get 1
i32.const 16
i32.add
global.set $__stack_pointer
local.get 4)
(func $prepend_alloc (type 2) (param i32 i32 i32) (result i32)
(local i32 i32 i32 i32 i32 i32 i32)
local.get 0
i32.const -8
local.get 0
i32.sub
i32.const 15
i32.and
i32.add
local.tee 3
local.get 2
i32.const 3
i32.or
i32.store offset=4
local.get 1
i32.const -8
local.get 1
i32.sub
i32.const 15
i32.and
i32.add
local.tee 4
local.get 3
local.get 2
i32.add
local.tee 5
i32.sub
local.set 2
block  ;; label = @1
    block  ;; label = @2
    local.get 4
    i32.const 0
    i32.load offset=82764
    i32.ne
    br_if 0 (;@2;)
    i32.const 0
    local.get 5
    i32.store offset=82764
    i32.const 0
    i32.const 0
    i32.load offset=82752
    local.get 2
    i32.add
    local.tee 2
    i32.store offset=82752
    local.get 5
    local.get 2
    i32.const 1
    i32.or
    i32.store offset=4
    br 1 (;@1;)
    end
    block  ;; label = @2
    local.get 4
    i32.const 0
    i32.load offset=82760
    i32.ne
    br_if 0 (;@2;)
    i32.const 0
    local.get 5
    i32.store offset=82760
    i32.const 0
    i32.const 0
    i32.load offset=82748
    local.get 2
    i32.add
    local.tee 2
    i32.store offset=82748
    local.get 5
    local.get 2
    i32.const 1
    i32.or
    i32.store offset=4
    local.get 5
    local.get 2
    i32.add
    local.get 2
    i32.store
    br 1 (;@1;)
    end
    block  ;; label = @2
    local.get 4
    i32.load offset=4
    local.tee 0
    i32.const 3
    i32.and
    i32.const 1
    i32.ne
    br_if 0 (;@2;)
    local.get 0
    i32.const -8
    i32.and
    local.set 6
    block  ;; label = @3
        block  ;; label = @4
        local.get 0
        i32.const 255
        i32.gt_u
        br_if 0 (;@4;)
        local.get 4
        i32.load offset=8
        local.tee 1
        local.get 0
        i32.const 3
        i32.shr_u
        local.tee 7
        i32.const 3
        i32.shl
        i32.const 82780
        i32.add
        local.tee 8
        i32.eq
        drop
        block  ;; label = @5
            local.get 4
            i32.load offset=12
            local.tee 0
            local.get 1
            i32.ne
            br_if 0 (;@5;)
            i32.const 0
            i32.const 0
            i32.load offset=82740
            i32.const -2
            local.get 7
            i32.rotl
            i32.and
            i32.store offset=82740
            br 2 (;@3;)
        end
        local.get 0
        local.get 8
        i32.eq
        drop
        local.get 0
        local.get 1
        i32.store offset=8
        local.get 1
        local.get 0
        i32.store offset=12
        br 1 (;@3;)
        end
        local.get 4
        i32.load offset=24
        local.set 9
        block  ;; label = @4
        block  ;; label = @5
            local.get 4
            i32.load offset=12
            local.tee 8
            local.get 4
            i32.eq
            br_if 0 (;@5;)
            local.get 4
            i32.load offset=8
            local.tee 0
            i32.const 0
            i32.load offset=82756
            i32.lt_u
            drop
            local.get 8
            local.get 0
            i32.store offset=8
            local.get 0
            local.get 8
            i32.store offset=12
            br 1 (;@4;)
        end
        block  ;; label = @5
            block  ;; label = @6
            local.get 4
            i32.const 20
            i32.add
            local.tee 1
            i32.load
            local.tee 0
            br_if 0 (;@6;)
            local.get 4
            i32.load offset=16
            local.tee 0
            i32.eqz
            br_if 1 (;@5;)
            local.get 4
            i32.const 16
            i32.add
            local.set 1
            end
            loop  ;; label = @6
            local.get 1
            local.set 7
            local.get 0
            local.tee 8
            i32.const 20
            i32.add
            local.tee 1
            i32.load
            local.tee 0
            br_if 0 (;@6;)
            local.get 8
            i32.const 16
            i32.add
            local.set 1
            local.get 8
            i32.load offset=16
            local.tee 0
            br_if 0 (;@6;)
            end
            local.get 7
            i32.const 0
            i32.store
            br 1 (;@4;)
        end
        i32.const 0
        local.set 8
        end
        local.get 9
        i32.eqz
        br_if 0 (;@3;)
        block  ;; label = @4
        block  ;; label = @5
            local.get 4
            local.get 4
            i32.load offset=28
            local.tee 1
            i32.const 2
            i32.shl
            i32.const 83044
            i32.add
            local.tee 0
            i32.load
            i32.ne
            br_if 0 (;@5;)
            local.get 0
            local.get 8
            i32.store
            local.get 8
            br_if 1 (;@4;)
            i32.const 0
            i32.const 0
            i32.load offset=82744
            i32.const -2
            local.get 1
            i32.rotl
            i32.and
            i32.store offset=82744
            br 2 (;@3;)
        end
        local.get 9
        i32.const 16
        i32.const 20
        local.get 9
        i32.load offset=16
        local.get 4
        i32.eq
        select
        i32.add
        local.get 8
        i32.store
        local.get 8
        i32.eqz
        br_if 1 (;@3;)
        end
        local.get 8
        local.get 9
        i32.store offset=24
        block  ;; label = @4
        local.get 4
        i32.load offset=16
        local.tee 0
        i32.eqz
        br_if 0 (;@4;)
        local.get 8
        local.get 0
        i32.store offset=16
        local.get 0
        local.get 8
        i32.store offset=24
        end
        local.get 4
        i32.const 20
        i32.add
        i32.load
        local.tee 0
        i32.eqz
        br_if 0 (;@3;)
        local.get 8
        i32.const 20
        i32.add
        local.get 0
        i32.store
        local.get 0
        local.get 8
        i32.store offset=24
    end
    local.get 6
    local.get 2
    i32.add
    local.set 2
    local.get 4
    local.get 6
    i32.add
    local.tee 4
    i32.load offset=4
    local.set 0
    end
    local.get 4
    local.get 0
    i32.const -2
    i32.and
    i32.store offset=4
    local.get 5
    local.get 2
    i32.add
    local.get 2
    i32.store
    local.get 5
    local.get 2
    i32.const 1
    i32.or
    i32.store offset=4
    block  ;; label = @2
    local.get 2
    i32.const 255
    i32.gt_u
    br_if 0 (;@2;)
    local.get 2
    i32.const -8
    i32.and
    i32.const 82780
    i32.add
    local.set 0
    block  ;; label = @3
        block  ;; label = @4
        i32.const 0
        i32.load offset=82740
        local.tee 1
        i32.const 1
        local.get 2
        i32.const 3
        i32.shr_u
        i32.shl
        local.tee 2
        i32.and
        br_if 0 (;@4;)
        i32.const 0
        local.get 1
        local.get 2
        i32.or
        i32.store offset=82740
        local.get 0
        local.set 2
        br 1 (;@3;)
        end
        local.get 0
        i32.load offset=8
        local.set 2
    end
    local.get 2
    local.get 5
    i32.store offset=12
    local.get 0
    local.get 5
    i32.store offset=8
    local.get 5
    local.get 0
    i32.store offset=12
    local.get 5
    local.get 2
    i32.store offset=8
    br 1 (;@1;)
    end
    i32.const 31
    local.set 0
    block  ;; label = @2
    local.get 2
    i32.const 16777215
    i32.gt_u
    br_if 0 (;@2;)
    local.get 2
    i32.const 38
    local.get 2
    i32.const 8
    i32.shr_u
    i32.clz
    local.tee 0
    i32.sub
    i32.shr_u
    i32.const 1
    i32.and
    local.get 0
    i32.const 1
    i32.shl
    i32.sub
    i32.const 62
    i32.add
    local.set 0
    end
    local.get 5
    local.get 0
    i32.store offset=28
    local.get 5
    i64.const 0
    i64.store offset=16 align=4
    local.get 0
    i32.const 2
    i32.shl
    i32.const 83044
    i32.add
    local.set 1
    block  ;; label = @2
    i32.const 0
    i32.load offset=82744
    local.tee 8
    i32.const 1
    local.get 0
    i32.shl
    local.tee 4
    i32.and
    br_if 0 (;@2;)
    local.get 1
    local.get 5
    i32.store
    i32.const 0
    local.get 8
    local.get 4
    i32.or
    i32.store offset=82744
    local.get 5
    local.get 1
    i32.store offset=24
    local.get 5
    local.get 5
    i32.store offset=8
    local.get 5
    local.get 5
    i32.store offset=12
    br 1 (;@1;)
    end
    local.get 2
    i32.const 0
    i32.const 25
    local.get 0
    i32.const 1
    i32.shr_u
    i32.sub
    local.get 0
    i32.const 31
    i32.eq
    select
    i32.shl
    local.set 0
    local.get 1
    i32.load
    local.set 8
    block  ;; label = @2
    loop  ;; label = @3
        local.get 8
        local.tee 1
        i32.load offset=4
        i32.const -8
        i32.and
        local.get 2
        i32.eq
        br_if 1 (;@2;)
        local.get 0
        i32.const 29
        i32.shr_u
        local.set 8
        local.get 0
        i32.const 1
        i32.shl
        local.set 0
        local.get 1
        local.get 8
        i32.const 4
        i32.and
        i32.add
        i32.const 16
        i32.add
        local.tee 4
        i32.load
        local.tee 8
        br_if 0 (;@3;)
    end
    local.get 4
    local.get 5
    i32.store
    local.get 5
    local.get 1
    i32.store offset=24
    local.get 5
    local.get 5
    i32.store offset=12
    local.get 5
    local.get 5
    i32.store offset=8
    br 1 (;@1;)
    end
    local.get 1
    i32.load offset=8
    local.tee 2
    local.get 5
    i32.store offset=12
    local.get 1
    local.get 5
    i32.store offset=8
    local.get 5
    i32.const 0
    i32.store offset=24
    local.get 5
    local.get 1
    i32.store offset=12
    local.get 5
    local.get 2
    i32.store offset=8
end
local.get 3
i32.const 8
i32.add)
(func $free (type 1) (param i32)
local.get 0
call $dlfree)
(func $dlfree (type 1) (param i32)
(local i32 i32 i32 i32 i32 i32 i32)
block  ;; label = @1
    local.get 0
    i32.eqz
    br_if 0 (;@1;)
    local.get 0
    i32.const -8
    i32.add
    local.tee 1
    local.get 0
    i32.const -4
    i32.add
    i32.load
    local.tee 2
    i32.const -8
    i32.and
    local.tee 0
    i32.add
    local.set 3
    block  ;; label = @2
    local.get 2
    i32.const 1
    i32.and
    br_if 0 (;@2;)
    local.get 2
    i32.const 2
    i32.and
    i32.eqz
    br_if 1 (;@1;)
    local.get 1
    local.get 1
    i32.load
    local.tee 2
    i32.sub
    local.tee 1
    i32.const 0
    i32.load offset=82756
    local.tee 4
    i32.lt_u
    br_if 1 (;@1;)
    local.get 2
    local.get 0
    i32.add
    local.set 0
    block  ;; label = @3
        block  ;; label = @4
        block  ;; label = @5
            local.get 1
            i32.const 0
            i32.load offset=82760
            i32.eq
            br_if 0 (;@5;)
            block  ;; label = @6
            local.get 2
            i32.const 255
            i32.gt_u
            br_if 0 (;@6;)
            local.get 1
            i32.load offset=8
            local.tee 4
            local.get 2
            i32.const 3
            i32.shr_u
            local.tee 5
            i32.const 3
            i32.shl
            i32.const 82780
            i32.add
            local.tee 6
            i32.eq
            drop
            block  ;; label = @7
                local.get 1
                i32.load offset=12
                local.tee 2
                local.get 4
                i32.ne
                br_if 0 (;@7;)
                i32.const 0
                i32.const 0
                i32.load offset=82740
                i32.const -2
                local.get 5
                i32.rotl
                i32.and
                i32.store offset=82740
                br 5 (;@2;)
            end
            local.get 2
            local.get 6
            i32.eq
            drop
            local.get 2
            local.get 4
            i32.store offset=8
            local.get 4
            local.get 2
            i32.store offset=12
            br 4 (;@2;)
            end
            local.get 1
            i32.load offset=24
            local.set 7
            block  ;; label = @6
            local.get 1
            i32.load offset=12
            local.tee 6
            local.get 1
            i32.eq
            br_if 0 (;@6;)
            local.get 1
            i32.load offset=8
            local.tee 2
            local.get 4
            i32.lt_u
            drop
            local.get 6
            local.get 2
            i32.store offset=8
            local.get 2
            local.get 6
            i32.store offset=12
            br 3 (;@3;)
            end
            block  ;; label = @6
            local.get 1
            i32.const 20
            i32.add
            local.tee 4
            i32.load
            local.tee 2
            br_if 0 (;@6;)
            local.get 1
            i32.load offset=16
            local.tee 2
            i32.eqz
            br_if 2 (;@4;)
            local.get 1
            i32.const 16
            i32.add
            local.set 4
            end
            loop  ;; label = @6
            local.get 4
            local.set 5
            local.get 2
            local.tee 6
            i32.const 20
            i32.add
            local.tee 4
            i32.load
            local.tee 2
            br_if 0 (;@6;)
            local.get 6
            i32.const 16
            i32.add
            local.set 4
            local.get 6
            i32.load offset=16
            local.tee 2
            br_if 0 (;@6;)
            end
            local.get 5
            i32.const 0
            i32.store
            br 2 (;@3;)
        end
        local.get 3
        i32.load offset=4
        local.tee 2
        i32.const 3
        i32.and
        i32.const 3
        i32.ne
        br_if 2 (;@2;)
        local.get 3
        local.get 2
        i32.const -2
        i32.and
        i32.store offset=4
        i32.const 0
        local.get 0
        i32.store offset=82748
        local.get 3
        local.get 0
        i32.store
        local.get 1
        local.get 0
        i32.const 1
        i32.or
        i32.store offset=4
        return
        end
        i32.const 0
        local.set 6
    end
    local.get 7
    i32.eqz
    br_if 0 (;@2;)
    block  ;; label = @3
        block  ;; label = @4
        local.get 1
        local.get 1
        i32.load offset=28
        local.tee 4
        i32.const 2
        i32.shl
        i32.const 83044
        i32.add
        local.tee 2
        i32.load
        i32.ne
        br_if 0 (;@4;)
        local.get 2
        local.get 6
        i32.store
        local.get 6
        br_if 1 (;@3;)
        i32.const 0
        i32.const 0
        i32.load offset=82744
        i32.const -2
        local.get 4
        i32.rotl
        i32.and
        i32.store offset=82744
        br 2 (;@2;)
        end
        local.get 7
        i32.const 16
        i32.const 20
        local.get 7
        i32.load offset=16
        local.get 1
        i32.eq
        select
        i32.add
        local.get 6
        i32.store
        local.get 6
        i32.eqz
        br_if 1 (;@2;)
    end
    local.get 6
    local.get 7
    i32.store offset=24
    block  ;; label = @3
        local.get 1
        i32.load offset=16
        local.tee 2
        i32.eqz
        br_if 0 (;@3;)
        local.get 6
        local.get 2
        i32.store offset=16
        local.get 2
        local.get 6
        i32.store offset=24
    end
    local.get 1
    i32.const 20
    i32.add
    i32.load
    local.tee 2
    i32.eqz
    br_if 0 (;@2;)
    local.get 6
    i32.const 20
    i32.add
    local.get 2
    i32.store
    local.get 2
    local.get 6
    i32.store offset=24
    end
    local.get 1
    local.get 3
    i32.ge_u
    br_if 0 (;@1;)
    local.get 3
    i32.load offset=4
    local.tee 2
    i32.const 1
    i32.and
    i32.eqz
    br_if 0 (;@1;)
    block  ;; label = @2
    block  ;; label = @3
        block  ;; label = @4
        block  ;; label = @5
            block  ;; label = @6
            local.get 2
            i32.const 2
            i32.and
            br_if 0 (;@6;)
            block  ;; label = @7
                local.get 3
                i32.const 0
                i32.load offset=82764
                i32.ne
                br_if 0 (;@7;)
                i32.const 0
                local.get 1
                i32.store offset=82764
                i32.const 0
                i32.const 0
                i32.load offset=82752
                local.get 0
                i32.add
                local.tee 0
                i32.store offset=82752
                local.get 1
                local.get 0
                i32.const 1
                i32.or
                i32.store offset=4
                local.get 1
                i32.const 0
                i32.load offset=82760
                i32.ne
                br_if 6 (;@1;)
                i32.const 0
                i32.const 0
                i32.store offset=82748
                i32.const 0
                i32.const 0
                i32.store offset=82760
                return
            end
            block  ;; label = @7
                local.get 3
                i32.const 0
                i32.load offset=82760
                i32.ne
                br_if 0 (;@7;)
                i32.const 0
                local.get 1
                i32.store offset=82760
                i32.const 0
                i32.const 0
                i32.load offset=82748
                local.get 0
                i32.add
                local.tee 0
                i32.store offset=82748
                local.get 1
                local.get 0
                i32.const 1
                i32.or
                i32.store offset=4
                local.get 1
                local.get 0
                i32.add
                local.get 0
                i32.store
                return
            end
            local.get 2
            i32.const -8
            i32.and
            local.get 0
            i32.add
            local.set 0
            block  ;; label = @7
                local.get 2
                i32.const 255
                i32.gt_u
                br_if 0 (;@7;)
                local.get 3
                i32.load offset=8
                local.tee 4
                local.get 2
                i32.const 3
                i32.shr_u
                local.tee 5
                i32.const 3
                i32.shl
                i32.const 82780
                i32.add
                local.tee 6
                i32.eq
                drop
                block  ;; label = @8
                local.get 3
                i32.load offset=12
                local.tee 2
                local.get 4
                i32.ne
                br_if 0 (;@8;)
                i32.const 0
                i32.const 0
                i32.load offset=82740
                i32.const -2
                local.get 5
                i32.rotl
                i32.and
                i32.store offset=82740
                br 5 (;@3;)
                end
                local.get 2
                local.get 6
                i32.eq
                drop
                local.get 2
                local.get 4
                i32.store offset=8
                local.get 4
                local.get 2
                i32.store offset=12
                br 4 (;@3;)
            end
            local.get 3
            i32.load offset=24
            local.set 7
            block  ;; label = @7
                local.get 3
                i32.load offset=12
                local.tee 6
                local.get 3
                i32.eq
                br_if 0 (;@7;)
                local.get 3
                i32.load offset=8
                local.tee 2
                i32.const 0
                i32.load offset=82756
                i32.lt_u
                drop
                local.get 6
                local.get 2
                i32.store offset=8
                local.get 2
                local.get 6
                i32.store offset=12
                br 3 (;@4;)
            end
            block  ;; label = @7
                local.get 3
                i32.const 20
                i32.add
                local.tee 4
                i32.load
                local.tee 2
                br_if 0 (;@7;)
                local.get 3
                i32.load offset=16
                local.tee 2
                i32.eqz
                br_if 2 (;@5;)
                local.get 3
                i32.const 16
                i32.add
                local.set 4
            end
            loop  ;; label = @7
                local.get 4
                local.set 5
                local.get 2
                local.tee 6
                i32.const 20
                i32.add
                local.tee 4
                i32.load
                local.tee 2
                br_if 0 (;@7;)
                local.get 6
                i32.const 16
                i32.add
                local.set 4
                local.get 6
                i32.load offset=16
                local.tee 2
                br_if 0 (;@7;)
            end
            local.get 5
            i32.const 0
            i32.store
            br 2 (;@4;)
            end
            local.get 3
            local.get 2
            i32.const -2
            i32.and
            i32.store offset=4
            local.get 1
            local.get 0
            i32.add
            local.get 0
            i32.store
            local.get 1
            local.get 0
            i32.const 1
            i32.or
            i32.store offset=4
            br 3 (;@2;)
        end
        i32.const 0
        local.set 6
        end
        local.get 7
        i32.eqz
        br_if 0 (;@3;)
        block  ;; label = @4
        block  ;; label = @5
            local.get 3
            local.get 3
            i32.load offset=28
            local.tee 4
            i32.const 2
            i32.shl
            i32.const 83044
            i32.add
            local.tee 2
            i32.load
            i32.ne
            br_if 0 (;@5;)
            local.get 2
            local.get 6
            i32.store
            local.get 6
            br_if 1 (;@4;)
            i32.const 0
            i32.const 0
            i32.load offset=82744
            i32.const -2
            local.get 4
            i32.rotl
            i32.and
            i32.store offset=82744
            br 2 (;@3;)
        end
        local.get 7
        i32.const 16
        i32.const 20
        local.get 7
        i32.load offset=16
        local.get 3
        i32.eq
        select
        i32.add
        local.get 6
        i32.store
        local.get 6
        i32.eqz
        br_if 1 (;@3;)
        end
        local.get 6
        local.get 7
        i32.store offset=24
        block  ;; label = @4
        local.get 3
        i32.load offset=16
        local.tee 2
        i32.eqz
        br_if 0 (;@4;)
        local.get 6
        local.get 2
        i32.store offset=16
        local.get 2
        local.get 6
        i32.store offset=24
        end
        local.get 3
        i32.const 20
        i32.add
        i32.load
        local.tee 2
        i32.eqz
        br_if 0 (;@3;)
        local.get 6
        i32.const 20
        i32.add
        local.get 2
        i32.store
        local.get 2
        local.get 6
        i32.store offset=24
    end
    local.get 1
    local.get 0
    i32.add
    local.get 0
    i32.store
    local.get 1
    local.get 0
    i32.const 1
    i32.or
    i32.store offset=4
    local.get 1
    i32.const 0
    i32.load offset=82760
    i32.ne
    br_if 0 (;@2;)
    i32.const 0
    local.get 0
    i32.store offset=82748
    return
    end
    block  ;; label = @2
    local.get 0
    i32.const 255
    i32.gt_u
    br_if 0 (;@2;)
    local.get 0
    i32.const -8
    i32.and
    i32.const 82780
    i32.add
    local.set 2
    block  ;; label = @3
        block  ;; label = @4
        i32.const 0
        i32.load offset=82740
        local.tee 4
        i32.const 1
        local.get 0
        i32.const 3
        i32.shr_u
        i32.shl
        local.tee 0
        i32.and
        br_if 0 (;@4;)
        i32.const 0
        local.get 4
        local.get 0
        i32.or
        i32.store offset=82740
        local.get 2
        local.set 0
        br 1 (;@3;)
        end
        local.get 2
        i32.load offset=8
        local.set 0
    end
    local.get 0
    local.get 1
    i32.store offset=12
    local.get 2
    local.get 1
    i32.store offset=8
    local.get 1
    local.get 2
    i32.store offset=12
    local.get 1
    local.get 0
    i32.store offset=8
    return
    end
    i32.const 31
    local.set 2
    block  ;; label = @2
    local.get 0
    i32.const 16777215
    i32.gt_u
    br_if 0 (;@2;)
    local.get 0
    i32.const 38
    local.get 0
    i32.const 8
    i32.shr_u
    i32.clz
    local.tee 2
    i32.sub
    i32.shr_u
    i32.const 1
    i32.and
    local.get 2
    i32.const 1
    i32.shl
    i32.sub
    i32.const 62
    i32.add
    local.set 2
    end
    local.get 1
    local.get 2
    i32.store offset=28
    local.get 1
    i64.const 0
    i64.store offset=16 align=4
    local.get 2
    i32.const 2
    i32.shl
    i32.const 83044
    i32.add
    local.set 4
    block  ;; label = @2
    block  ;; label = @3
        i32.const 0
        i32.load offset=82744
        local.tee 6
        i32.const 1
        local.get 2
        i32.shl
        local.tee 3
        i32.and
        br_if 0 (;@3;)
        local.get 4
        local.get 1
        i32.store
        i32.const 0
        local.get 6
        local.get 3
        i32.or
        i32.store offset=82744
        local.get 1
        local.get 4
        i32.store offset=24
        local.get 1
        local.get 1
        i32.store offset=8
        local.get 1
        local.get 1
        i32.store offset=12
        br 1 (;@2;)
    end
    local.get 0
    i32.const 0
    i32.const 25
    local.get 2
    i32.const 1
    i32.shr_u
    i32.sub
    local.get 2
    i32.const 31
    i32.eq
    select
    i32.shl
    local.set 2
    local.get 4
    i32.load
    local.set 6
    block  ;; label = @3
        loop  ;; label = @4
        local.get 6
        local.tee 4
        i32.load offset=4
        i32.const -8
        i32.and
        local.get 0
        i32.eq
        br_if 1 (;@3;)
        local.get 2
        i32.const 29
        i32.shr_u
        local.set 6
        local.get 2
        i32.const 1
        i32.shl
        local.set 2
        local.get 4
        local.get 6
        i32.const 4
        i32.and
        i32.add
        i32.const 16
        i32.add
        local.tee 3
        i32.load
        local.tee 6
        br_if 0 (;@4;)
        end
        local.get 3
        local.get 1
        i32.store
        local.get 1
        local.get 4
        i32.store offset=24
        local.get 1
        local.get 1
        i32.store offset=12
        local.get 1
        local.get 1
        i32.store offset=8
        br 1 (;@2;)
    end
    local.get 4
    i32.load offset=8
    local.tee 0
    local.get 1
    i32.store offset=12
    local.get 4
    local.get 1
    i32.store offset=8
    local.get 1
    i32.const 0
    i32.store offset=24
    local.get 1
    local.get 4
    i32.store offset=12
    local.get 1
    local.get 0
    i32.store offset=8
    end
    i32.const 0
    i32.const 0
    i32.load offset=82772
    i32.const -1
    i32.add
    local.tee 1
    i32.const -1
    local.get 1
    select
    i32.store offset=82772
end)
;; end of malloc/free functions
