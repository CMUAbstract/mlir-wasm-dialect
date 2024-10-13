  ;; unoptimized wat for malloc and free
  (global (;0;) (mut i32) (i32.const 98496)) ;; stack pointer
  (func $helper (type 0) (param i32) (result i32)
    local.get 0
    i32.eqz
    if  ;; label = @1
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
      i32.const 0
      i32.lt_s
      br_if 0 (;@1;)
      local.get 0
      i32.const 16
      i32.shr_u
      memory.grow
      local.tee 0
      i32.const -1
      i32.eq
      if  ;; label = @2
        i32.const 32448
        i32.const 48
        i32.store
        i32.const -1
        return
      end
      local.get 0
      i32.const 16
      i32.shl
      return
    end
    unreachable)
  (func $malloc (type 0) (param i32) (result i32)
    (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
    local.get 0
    local.set 1
    i32.const 0
    local.set 0
    global.get 0
    i32.const 16
    i32.sub
    local.tee 11
    global.set 0
    i32.const 32476
    i32.load
    local.tee 9
    i32.eqz
    if  ;; label = @1
      i32.const 32924
      i32.load
      local.tee 6
      i32.eqz
      if  ;; label = @2
        i32.const 32936
        i64.const -1
        i64.store align=4
        i32.const 32928
        i64.const 281474976776192
        i64.store align=4
        i32.const 32924
        local.get 11
        i32.const 8
        i32.add
        i32.const -16
        i32.and
        i32.const 1431655768
        i32.xor
        local.tee 6
        i32.store
        i32.const 32944
        i32.const 0
        i32.store
        i32.const 32896
        i32.const 0
        i32.store
      end
      i32.const 32900
      i32.const 98496
      i32.store
      i32.const 32468
      i32.const 98496
      i32.store
      i32.const 32488
      local.get 6
      i32.store
      i32.const 32484
      i32.const -1
      i32.store
      i32.const 32904
      i32.const 32576
      i32.store
      loop  ;; label = @2
        local.get 0
        i32.const 32512
        i32.add
        local.get 0
        i32.const 32500
        i32.add
        local.tee 7
        i32.store
        local.get 7
        local.get 0
        i32.const 32492
        i32.add
        local.tee 2
        i32.store
        local.get 0
        i32.const 32504
        i32.add
        local.get 2
        i32.store
        local.get 0
        i32.const 32520
        i32.add
        local.get 0
        i32.const 32508
        i32.add
        local.tee 3
        i32.store
        local.get 3
        local.get 7
        i32.store
        local.get 0
        i32.const 32528
        i32.add
        local.get 0
        i32.const 32516
        i32.add
        local.tee 2
        i32.store
        local.get 2
        local.get 3
        i32.store
        local.get 0
        i32.const 32524
        i32.add
        local.get 2
        i32.store
        local.get 0
        i32.const 32
        i32.add
        local.tee 0
        i32.const 256
        i32.ne
        br_if 0 (;@2;)
      end
      i32.const 98508
      i32.const 32513
      i32.store
      i32.const 32480
      i32.const 32940
      i32.load
      i32.store
      i32.const 32464
      i32.const 32512
      i32.store
      i32.const 32476
      i32.const 98504
      i32.store
      i32.const 131020
      i32.const 56
      i32.store
      i32.const 98504
      local.set 9
    end
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
                          local.get 1
                          i32.const 236
                          i32.le_u
                          if  ;; label = @12
                            i32.const 32452
                            i32.load
                            local.tee 8
                            i32.const 16
                            local.get 1
                            i32.const 19
                            i32.add
                            i32.const 496
                            i32.and
                            local.get 1
                            i32.const 11
                            i32.lt_u
                            select
                            local.tee 7
                            i32.const 3
                            i32.shr_u
                            local.tee 2
                            i32.shr_u
                            local.tee 1
                            i32.const 3
                            i32.and
                            if  ;; label = @13
                              block  ;; label = @14
                                local.get 1
                                i32.const 1
                                i32.and
                                local.get 2
                                i32.or
                                i32.const 1
                                i32.xor
                                local.tee 2
                                i32.const 3
                                i32.shl
                                local.tee 0
                                i32.const 32492
                                i32.add
                                local.tee 1
                                local.get 0
                                i32.const 32500
                                i32.add
                                i32.load
                                local.tee 3
                                i32.load offset=8
                                local.tee 0
                                i32.eq
                                if  ;; label = @15
                                  i32.const 32452
                                  local.get 8
                                  i32.const -2
                                  local.get 2
                                  i32.rotl
                                  i32.and
                                  i32.store
                                  br 1 (;@14;)
                                end
                                local.get 1
                                local.get 0
                                i32.store offset=8
                                local.get 0
                                local.get 1
                                i32.store offset=12
                              end
                              local.get 3
                              i32.const 8
                              i32.add
                              local.set 0
                              local.get 3
                              local.get 2
                              i32.const 3
                              i32.shl
                              local.tee 1
                              i32.const 3
                              i32.or
                              i32.store offset=4
                              local.get 1
                              local.get 3
                              i32.add
                              local.tee 1
                              local.get 1
                              i32.load offset=4
                              i32.const 1
                              i32.or
                              i32.store offset=4
                              br 12 (;@1;)
                            end
                            i32.const 32460
                            i32.load
                            local.tee 10
                            local.get 7
                            i32.ge_u
                            br_if 1 (;@11;)
                            local.get 1
                            if  ;; label = @13
                              block  ;; label = @14
                                i32.const 0
                                i32.const 2
                                local.get 2
                                i32.shl
                                local.tee 0
                                i32.sub
                                local.get 0
                                i32.or
                                local.get 1
                                local.get 2
                                i32.shl
                                i32.and
                                i32.ctz
                                local.tee 2
                                i32.const 3
                                i32.shl
                                local.tee 0
                                i32.const 32492
                                i32.add
                                local.tee 1
                                local.get 0
                                i32.const 32500
                                i32.add
                                i32.load
                                local.tee 5
                                i32.load offset=8
                                local.tee 0
                                i32.eq
                                if  ;; label = @15
                                  i32.const 32452
                                  local.get 8
                                  i32.const -2
                                  local.get 2
                                  i32.rotl
                                  i32.and
                                  local.tee 8
                                  i32.store
                                  br 1 (;@14;)
                                end
                                local.get 1
                                local.get 0
                                i32.store offset=8
                                local.get 0
                                local.get 1
                                i32.store offset=12
                              end
                              local.get 5
                              local.get 7
                              i32.const 3
                              i32.or
                              i32.store offset=4
                              local.get 2
                              i32.const 3
                              i32.shl
                              local.tee 0
                              local.get 7
                              i32.sub
                              local.set 3
                              local.get 0
                              local.get 5
                              i32.add
                              local.get 3
                              i32.store
                              local.get 5
                              local.get 7
                              i32.add
                              local.tee 2
                              local.get 3
                              i32.const 1
                              i32.or
                              i32.store offset=4
                              local.get 10
                              if  ;; label = @14
                                local.get 10
                                i32.const -8
                                i32.and
                                i32.const 32492
                                i32.add
                                local.set 1
                                i32.const 32472
                                i32.load
                                local.set 7
                                block (result i32)  ;; label = @15
                                  i32.const 1
                                  local.get 10
                                  i32.const 3
                                  i32.shr_u
                                  i32.shl
                                  local.tee 0
                                  local.get 8
                                  i32.and
                                  i32.eqz
                                  if  ;; label = @16
                                    i32.const 32452
                                    local.get 0
                                    local.get 8
                                    i32.or
                                    i32.store
                                    local.get 1
                                    br 1 (;@15;)
                                  end
                                  local.get 1
                                  i32.load offset=8
                                end
                                local.tee 0
                                local.get 7
                                i32.store offset=12
                                local.get 1
                                local.get 7
                                i32.store offset=8
                                local.get 7
                                local.get 1
                                i32.store offset=12
                                local.get 7
                                local.get 0
                                i32.store offset=8
                              end
                              local.get 5
                              i32.const 8
                              i32.add
                              local.set 0
                              i32.const 32472
                              local.get 2
                              i32.store
                              i32.const 32460
                              local.get 3
                              i32.store
                              br 12 (;@1;)
                            end
                            i32.const 32456
                            i32.load
                            local.tee 4
                            i32.eqz
                            br_if 1 (;@11;)
                            local.get 4
                            i32.ctz
                            i32.const 2
                            i32.shl
                            i32.const 32756
                            i32.add
                            i32.load
                            local.tee 1
                            i32.load offset=4
                            i32.const -8
                            i32.and
                            local.get 7
                            i32.sub
                            local.set 6
                            local.get 1
                            local.set 2
                            loop  ;; label = @13
                              block  ;; label = @14
                                local.get 2
                                i32.load offset=16
                                local.tee 0
                                i32.eqz
                                if  ;; label = @15
                                  local.get 2
                                  i32.const 20
                                  i32.add
                                  i32.load
                                  local.tee 0
                                  i32.eqz
                                  br_if 1 (;@14;)
                                end
                                local.get 0
                                i32.load offset=4
                                i32.const -8
                                i32.and
                                local.get 7
                                i32.sub
                                local.tee 2
                                local.get 6
                                i32.lt_u
                                local.set 3
                                local.get 2
                                local.get 6
                                local.get 3
                                select
                                local.set 6
                                local.get 0
                                local.get 1
                                local.get 3
                                select
                                local.set 1
                                local.get 0
                                local.set 2
                                br 1 (;@13;)
                              end
                            end
                            local.get 1
                            i32.load offset=24
                            local.set 9
                            local.get 1
                            i32.load offset=12
                            local.tee 3
                            local.get 1
                            i32.ne
                            if  ;; label = @13
                              i32.const 32468
                              i32.load
                              drop
                              local.get 3
                              local.get 1
                              i32.load offset=8
                              local.tee 0
                              i32.store offset=8
                              local.get 0
                              local.get 3
                              i32.store offset=12
                              br 11 (;@2;)
                            end
                            local.get 1
                            i32.const 20
                            i32.add
                            local.tee 2
                            i32.load
                            local.tee 0
                            i32.eqz
                            if  ;; label = @13
                              local.get 1
                              i32.load offset=16
                              local.tee 0
                              i32.eqz
                              br_if 3 (;@10;)
                              local.get 1
                              i32.const 16
                              i32.add
                              local.set 2
                            end
                            loop  ;; label = @13
                              local.get 2
                              local.set 5
                              local.get 0
                              local.tee 3
                              i32.const 20
                              i32.add
                              local.tee 2
                              i32.load
                              local.tee 0
                              br_if 0 (;@13;)
                              local.get 3
                              i32.const 16
                              i32.add
                              local.set 2
                              local.get 3
                              i32.load offset=16
                              local.tee 0
                              br_if 0 (;@13;)
                            end
                            local.get 5
                            i32.const 0
                            i32.store
                            br 10 (;@2;)
                          end
                          i32.const -1
                          local.set 7
                          local.get 1
                          i32.const -65
                          i32.gt_u
                          br_if 0 (;@11;)
                          local.get 1
                          i32.const 19
                          i32.add
                          local.tee 0
                          i32.const -16
                          i32.and
                          local.set 7
                          i32.const 32456
                          i32.load
                          local.tee 4
                          i32.eqz
                          br_if 0 (;@11;)
                          i32.const 0
                          local.get 7
                          i32.sub
                          local.set 6
                          block  ;; label = @12
                            block  ;; label = @13
                              block  ;; label = @14
                                block (result i32)  ;; label = @15
                                  i32.const 0
                                  local.get 7
                                  i32.const 256
                                  i32.lt_u
                                  br_if 0 (;@15;)
                                  drop
                                  i32.const 31
                                  local.get 7
                                  i32.const 16777215
                                  i32.gt_u
                                  br_if 0 (;@15;)
                                  drop
                                  local.get 7
                                  i32.const 38
                                  local.get 0
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
                                end
                                local.tee 8
                                i32.const 2
                                i32.shl
                                i32.const 32756
                                i32.add
                                i32.load
                                local.tee 2
                                i32.eqz
                                if  ;; label = @15
                                  i32.const 0
                                  local.set 0
                                  i32.const 0
                                  local.set 3
                                  br 1 (;@14;)
                                end
                                i32.const 0
                                local.set 0
                                local.get 7
                                i32.const 25
                                local.get 8
                                i32.const 1
                                i32.shr_u
                                i32.sub
                                i32.const 0
                                local.get 8
                                i32.const 31
                                i32.ne
                                select
                                i32.shl
                                local.set 1
                                i32.const 0
                                local.set 3
                                loop  ;; label = @15
                                  block  ;; label = @16
                                    local.get 2
                                    i32.load offset=4
                                    i32.const -8
                                    i32.and
                                    local.get 7
                                    i32.sub
                                    local.tee 5
                                    local.get 6
                                    i32.ge_u
                                    br_if 0 (;@16;)
                                    local.get 2
                                    local.set 3
                                    local.get 5
                                    local.tee 6
                                    br_if 0 (;@16;)
                                    i32.const 0
                                    local.set 6
                                    local.get 3
                                    local.set 0
                                    br 3 (;@13;)
                                  end
                                  local.get 0
                                  local.get 2
                                  i32.const 20
                                  i32.add
                                  i32.load
                                  local.tee 5
                                  local.get 5
                                  local.get 2
                                  local.get 1
                                  i32.const 29
                                  i32.shr_u
                                  i32.const 4
                                  i32.and
                                  i32.add
                                  i32.const 16
                                  i32.add
                                  i32.load
                                  local.tee 2
                                  i32.eq
                                  select
                                  local.get 0
                                  local.get 5
                                  select
                                  local.set 0
                                  local.get 1
                                  i32.const 1
                                  i32.shl
                                  local.set 1
                                  local.get 2
                                  br_if 0 (;@15;)
                                end
                              end
                              local.get 0
                              local.get 3
                              i32.or
                              i32.eqz
                              if  ;; label = @14
                                i32.const 0
                                local.set 3
                                i32.const 0
                                i32.const 2
                                local.get 8
                                i32.shl
                                local.tee 0
                                i32.sub
                                local.get 0
                                i32.or
                                local.get 4
                                i32.and
                                local.tee 0
                                i32.eqz
                                br_if 3 (;@11;)
                                local.get 0
                                i32.ctz
                                i32.const 2
                                i32.shl
                                i32.const 32756
                                i32.add
                                i32.load
                                local.set 0
                              end
                              local.get 0
                              i32.eqz
                              br_if 1 (;@12;)
                            end
                            loop  ;; label = @13
                              local.get 0
                              i32.load offset=4
                              i32.const -8
                              i32.and
                              local.get 7
                              i32.sub
                              local.tee 1
                              local.get 6
                              i32.lt_u
                              local.set 2
                              local.get 1
                              local.get 6
                              local.get 2
                              select
                              local.set 6
                              local.get 0
                              local.get 3
                              local.get 2
                              select
                              local.set 3
                              local.get 0
                              i32.load offset=16
                              local.tee 1
                              if (result i32)  ;; label = @14
                                local.get 1
                              else
                                local.get 0
                                i32.const 20
                                i32.add
                                i32.load
                              end
                              local.tee 0
                              br_if 0 (;@13;)
                            end
                          end
                          local.get 3
                          i32.eqz
                          br_if 0 (;@11;)
                          local.get 6
                          i32.const 32460
                          i32.load
                          local.get 7
                          i32.sub
                          i32.ge_u
                          br_if 0 (;@11;)
                          local.get 3
                          i32.load offset=24
                          local.set 8
                          local.get 3
                          local.get 3
                          i32.load offset=12
                          local.tee 1
                          i32.ne
                          if  ;; label = @12
                            i32.const 32468
                            i32.load
                            drop
                            local.get 1
                            local.get 3
                            i32.load offset=8
                            local.tee 0
                            i32.store offset=8
                            local.get 0
                            local.get 1
                            i32.store offset=12
                            br 9 (;@3;)
                          end
                          local.get 3
                          i32.const 20
                          i32.add
                          local.tee 2
                          i32.load
                          local.tee 0
                          i32.eqz
                          if  ;; label = @12
                            local.get 3
                            i32.load offset=16
                            local.tee 0
                            i32.eqz
                            br_if 3 (;@9;)
                            local.get 3
                            i32.const 16
                            i32.add
                            local.set 2
                          end
                          loop  ;; label = @12
                            local.get 2
                            local.set 5
                            local.get 0
                            local.tee 1
                            i32.const 20
                            i32.add
                            local.tee 2
                            i32.load
                            local.tee 0
                            br_if 0 (;@12;)
                            local.get 1
                            i32.const 16
                            i32.add
                            local.set 2
                            local.get 1
                            i32.load offset=16
                            local.tee 0
                            br_if 0 (;@12;)
                          end
                          local.get 5
                          i32.const 0
                          i32.store
                          br 8 (;@3;)
                        end
                        i32.const 32460
                        i32.load
                        local.tee 0
                        local.get 7
                        i32.ge_u
                        if  ;; label = @11
                          i32.const 32472
                          i32.load
                          local.set 3
                          block  ;; label = @12
                            local.get 0
                            local.get 7
                            i32.sub
                            local.tee 2
                            i32.const 16
                            i32.ge_u
                            if  ;; label = @13
                              local.get 3
                              local.get 7
                              i32.add
                              local.tee 1
                              local.get 2
                              i32.const 1
                              i32.or
                              i32.store offset=4
                              local.get 0
                              local.get 3
                              i32.add
                              local.get 2
                              i32.store
                              local.get 3
                              local.get 7
                              i32.const 3
                              i32.or
                              i32.store offset=4
                              br 1 (;@12;)
                            end
                            local.get 3
                            local.get 0
                            i32.const 3
                            i32.or
                            i32.store offset=4
                            local.get 0
                            local.get 3
                            i32.add
                            local.tee 0
                            local.get 0
                            i32.load offset=4
                            i32.const 1
                            i32.or
                            i32.store offset=4
                            i32.const 0
                            local.set 1
                            i32.const 0
                            local.set 2
                          end
                          i32.const 32460
                          local.get 2
                          i32.store
                          i32.const 32472
                          local.get 1
                          i32.store
                          local.get 3
                          i32.const 8
                          i32.add
                          local.set 0
                          br 10 (;@1;)
                        end
                        i32.const 32464
                        i32.load
                        local.tee 4
                        local.get 7
                        i32.gt_u
                        if  ;; label = @11
                          local.get 7
                          local.get 9
                          i32.add
                          local.tee 1
                          local.get 4
                          local.get 7
                          i32.sub
                          local.tee 0
                          i32.const 1
                          i32.or
                          i32.store offset=4
                          i32.const 32476
                          local.get 1
                          i32.store
                          i32.const 32464
                          local.get 0
                          i32.store
                          local.get 9
                          local.get 7
                          i32.const 3
                          i32.or
                          i32.store offset=4
                          local.get 9
                          i32.const 8
                          i32.add
                          local.set 0
                          br 10 (;@1;)
                        end
                        i32.const 0
                        local.set 0
                        local.get 7
                        block (result i32)  ;; label = @11
                          i32.const 32924
                          i32.load
                          if  ;; label = @12
                            i32.const 32932
                            i32.load
                            br 1 (;@11;)
                          end
                          i32.const 32936
                          i64.const -1
                          i64.store align=4
                          i32.const 32928
                          i64.const 281474976776192
                          i64.store align=4
                          i32.const 32924
                          local.get 11
                          i32.const 12
                          i32.add
                          i32.const -16
                          i32.and
                          i32.const 1431655768
                          i32.xor
                          i32.store
                          i32.const 32944
                          i32.const 0
                          i32.store
                          i32.const 32896
                          i32.const 0
                          i32.store
                          i32.const 65536
                        end
                        local.tee 1
                        local.get 7
                        i32.const 71
                        i32.add
                        local.tee 6
                        i32.add
                        local.tee 5
                        i32.const 0
                        local.get 1
                        i32.sub
                        local.tee 3
                        i32.and
                        local.tee 2
                        i32.ge_u
                        if  ;; label = @11
                          i32.const 32448
                          i32.const 48
                          i32.store
                          br 10 (;@1;)
                        end
                        block  ;; label = @11
                          i32.const 32892
                          i32.load
                          local.tee 1
                          i32.eqz
                          br_if 0 (;@11;)
                          i32.const 32884
                          i32.load
                          local.tee 0
                          local.get 2
                          i32.add
                          local.set 8
                          local.get 0
                          local.get 8
                          i32.lt_u
                          local.get 1
                          local.get 8
                          i32.ge_u
                          i32.and
                          br_if 0 (;@11;)
                          i32.const 0
                          local.set 0
                          i32.const 32448
                          i32.const 48
                          i32.store
                          br 10 (;@1;)
                        end
                        i32.const 32896
                        i32.load8_u
                        i32.const 4
                        i32.and
                        br_if 4 (;@6;)
                        block  ;; label = @11
                          block  ;; label = @12
                            local.get 9
                            if  ;; label = @13
                              i32.const 32900
                              local.set 0
                              loop  ;; label = @14
                                local.get 0
                                i32.load
                                local.tee 1
                                local.get 9
                                i32.le_u
                                if  ;; label = @15
                                  local.get 1
                                  local.get 0
                                  i32.load offset=4
                                  i32.add
                                  local.get 9
                                  i32.gt_u
                                  br_if 3 (;@12;)
                                end
                                local.get 0
                                i32.load offset=8
                                local.tee 0
                                br_if 0 (;@14;)
                              end
                            end
                            i32.const 0
                            call $helper
                            local.tee 1
                            i32.const -1
                            i32.eq
                            br_if 5 (;@7;)
                            local.get 2
                            local.set 8
                            i32.const 32928
                            i32.load
                            local.tee 3
                            i32.const 1
                            i32.sub
                            local.tee 0
                            local.get 1
                            i32.and
                            if  ;; label = @13
                              local.get 2
                              local.get 1
                              i32.sub
                              local.get 0
                              local.get 1
                              i32.add
                              i32.const 0
                              local.get 3
                              i32.sub
                              i32.and
                              i32.add
                              local.set 8
                            end
                            local.get 7
                            local.get 8
                            i32.ge_u
                            br_if 5 (;@7;)
                            local.get 8
                            i32.const 2147483646
                            i32.gt_u
                            br_if 5 (;@7;)
                            i32.const 32892
                            i32.load
                            local.tee 3
                            if  ;; label = @13
                              i32.const 32884
                              i32.load
                              local.tee 0
                              local.get 8
                              i32.add
                              local.set 5
                              local.get 0
                              local.get 5
                              i32.ge_u
                              br_if 6 (;@7;)
                              local.get 3
                              local.get 5
                              i32.lt_u
                              br_if 6 (;@7;)
                            end
                            local.get 1
                            local.get 8
                            call $helper
                            local.tee 0
                            i32.ne
                            br_if 1 (;@11;)
                            br 7 (;@5;)
                          end
                          local.get 5
                          local.get 4
                          i32.sub
                          local.get 3
                          i32.and
                          local.tee 8
                          i32.const 2147483646
                          i32.gt_u
                          br_if 4 (;@7;)
                          local.get 8
                          call $helper
                          local.set 1
                          local.get 1
                          local.get 0
                          i32.load
                          local.get 0
                          i32.load offset=4
                          i32.add
                          i32.eq
                          br_if 3 (;@8;)
                          local.get 1
                          local.set 0
                        end
                        block  ;; label = @11
                          local.get 8
                          local.get 7
                          i32.const 72
                          i32.add
                          i32.ge_u
                          br_if 0 (;@11;)
                          local.get 0
                          i32.const -1
                          i32.eq
                          br_if 0 (;@11;)
                          i32.const 32932
                          i32.load
                          local.tee 1
                          local.get 6
                          local.get 8
                          i32.sub
                          i32.add
                          i32.const 0
                          local.get 1
                          i32.sub
                          i32.and
                          local.tee 1
                          i32.const 2147483646
                          i32.gt_u
                          if  ;; label = @12
                            local.get 0
                            local.set 1
                            br 7 (;@5;)
                          end
                          local.get 1
                          call $helper
                          i32.const -1
                          i32.ne
                          if  ;; label = @12
                            local.get 1
                            local.get 8
                            i32.add
                            local.set 8
                            local.get 0
                            local.set 1
                            br 7 (;@5;)
                          end
                          i32.const 0
                          local.get 8
                          i32.sub
                          call $helper
                          drop
                          br 4 (;@7;)
                        end
                        local.get 0
                        local.tee 1
                        i32.const -1
                        i32.ne
                        br_if 5 (;@5;)
                        br 3 (;@7;)
                      end
                      i32.const 0
                      local.set 3
                      br 7 (;@2;)
                    end
                    i32.const 0
                    local.set 1
                    br 5 (;@3;)
                  end
                  local.get 1
                  i32.const -1
                  i32.ne
                  br_if 2 (;@5;)
                end
                i32.const 32896
                i32.const 32896
                i32.load
                i32.const 4
                i32.or
                i32.store
              end
              local.get 2
              i32.const 2147483646
              i32.gt_u
              br_if 1 (;@4;)
              local.get 2
              call $helper
              local.set 1
              i32.const 0
              call $helper
              local.set 0
              local.get 1
              i32.const -1
              i32.eq
              br_if 1 (;@4;)
              local.get 0
              i32.const -1
              i32.eq
              br_if 1 (;@4;)
              local.get 0
              local.get 1
              i32.le_u
              br_if 1 (;@4;)
              local.get 0
              local.get 1
              i32.sub
              local.tee 8
              local.get 7
              i32.const 56
              i32.add
              i32.le_u
              br_if 1 (;@4;)
            end
            i32.const 32884
            i32.const 32884
            i32.load
            local.get 8
            i32.add
            local.tee 0
            i32.store
            i32.const 32888
            i32.load
            local.get 0
            i32.lt_u
            if  ;; label = @5
              i32.const 32888
              local.get 0
              i32.store
            end
            block  ;; label = @5
              block  ;; label = @6
                block  ;; label = @7
                  i32.const 32476
                  i32.load
                  local.tee 4
                  if  ;; label = @8
                    i32.const 32900
                    local.set 0
                    loop  ;; label = @9
                      local.get 1
                      local.get 0
                      i32.load
                      local.tee 2
                      local.get 0
                      i32.load offset=4
                      local.tee 5
                      i32.add
                      i32.eq
                      br_if 2 (;@7;)
                      local.get 0
                      i32.load offset=8
                      local.tee 0
                      br_if 0 (;@9;)
                    end
                    br 2 (;@6;)
                  end
                  i32.const 32468
                  i32.load
                  local.tee 0
                  i32.const 0
                  i32.ne
                  local.get 0
                  local.get 1
                  i32.le_u
                  i32.and
                  i32.eqz
                  if  ;; label = @8
                    i32.const 32468
                    local.get 1
                    i32.store
                  end
                  i32.const 0
                  local.set 0
                  i32.const 32904
                  local.get 8
                  i32.store
                  i32.const 32900
                  local.get 1
                  i32.store
                  i32.const 32484
                  i32.const -1
                  i32.store
                  i32.const 32488
                  i32.const 32924
                  i32.load
                  i32.store
                  i32.const 32912
                  i32.const 0
                  i32.store
                  loop  ;; label = @8
                    local.get 0
                    i32.const 32512
                    i32.add
                    local.get 0
                    i32.const 32500
                    i32.add
                    local.tee 5
                    i32.store
                    local.get 5
                    local.get 0
                    i32.const 32492
                    i32.add
                    local.tee 2
                    i32.store
                    local.get 0
                    i32.const 32504
                    i32.add
                    local.get 2
                    i32.store
                    local.get 0
                    i32.const 32520
                    i32.add
                    local.get 0
                    i32.const 32508
                    i32.add
                    local.tee 3
                    i32.store
                    local.get 3
                    local.get 5
                    i32.store
                    local.get 0
                    i32.const 32528
                    i32.add
                    local.get 0
                    i32.const 32516
                    i32.add
                    local.tee 2
                    i32.store
                    local.get 2
                    local.get 3
                    i32.store
                    local.get 0
                    i32.const 32524
                    i32.add
                    local.get 2
                    i32.store
                    local.get 0
                    i32.const 32
                    i32.add
                    local.tee 0
                    i32.const 256
                    i32.ne
                    br_if 0 (;@8;)
                  end
                  i32.const -8
                  local.get 1
                  i32.sub
                  i32.const 15
                  i32.and
                  local.tee 0
                  local.get 1
                  i32.add
                  local.tee 3
                  local.get 8
                  i32.const 56
                  i32.sub
                  local.tee 2
                  local.get 0
                  i32.sub
                  local.tee 0
                  i32.const 1
                  i32.or
                  i32.store offset=4
                  i32.const 32480
                  i32.const 32940
                  i32.load
                  i32.store
                  i32.const 32464
                  local.get 0
                  i32.store
                  i32.const 32476
                  local.get 3
                  i32.store
                  local.get 1
                  local.get 2
                  i32.add
                  i32.const 56
                  i32.store offset=4
                  br 2 (;@5;)
                end
                local.get 1
                local.get 4
                i32.le_u
                br_if 0 (;@6;)
                local.get 2
                local.get 4
                i32.gt_u
                br_if 0 (;@6;)
                local.get 0
                i32.load offset=12
                i32.const 8
                i32.and
                br_if 0 (;@6;)
                i32.const -8
                local.get 4
                i32.sub
                i32.const 15
                i32.and
                local.tee 1
                local.get 4
                i32.add
                local.tee 3
                i32.const 32464
                i32.load
                local.get 8
                i32.add
                local.tee 2
                local.get 1
                i32.sub
                local.tee 1
                i32.const 1
                i32.or
                i32.store offset=4
                local.get 0
                local.get 5
                local.get 8
                i32.add
                i32.store offset=4
                i32.const 32480
                i32.const 32940
                i32.load
                i32.store
                i32.const 32464
                local.get 1
                i32.store
                i32.const 32476
                local.get 3
                i32.store
                local.get 2
                local.get 4
                i32.add
                i32.const 56
                i32.store offset=4
                br 1 (;@5;)
              end
              i32.const 32468
              i32.load
              local.get 1
              i32.gt_u
              if  ;; label = @6
                i32.const 32468
                local.get 1
                i32.store
              end
              local.get 1
              local.get 8
              i32.add
              local.set 2
              i32.const 32900
              local.set 0
              block  ;; label = @6
                block  ;; label = @7
                  block  ;; label = @8
                    loop  ;; label = @9
                      local.get 2
                      local.get 0
                      i32.load
                      i32.ne
                      if  ;; label = @10
                        local.get 0
                        i32.load offset=8
                        local.tee 0
                        br_if 1 (;@9;)
                        br 2 (;@8;)
                      end
                    end
                    local.get 0
                    i32.load8_u offset=12
                    i32.const 8
                    i32.and
                    i32.eqz
                    br_if 1 (;@7;)
                  end
                  i32.const 32900
                  local.set 0
                  loop  ;; label = @8
                    block  ;; label = @9
                      local.get 0
                      i32.load
                      local.tee 2
                      local.get 4
                      i32.le_u
                      if  ;; label = @10
                        local.get 2
                        local.get 0
                        i32.load offset=4
                        i32.add
                        local.tee 5
                        local.get 4
                        i32.gt_u
                        br_if 1 (;@9;)
                      end
                      local.get 0
                      i32.load offset=8
                      local.set 0
                      br 1 (;@8;)
                    end
                  end
                  i32.const -8
                  local.get 1
                  i32.sub
                  i32.const 15
                  i32.and
                  local.tee 2
                  local.get 1
                  i32.add
                  local.tee 3
                  local.get 8
                  i32.const 56
                  i32.sub
                  local.tee 0
                  local.get 2
                  i32.sub
                  local.tee 2
                  i32.const 1
                  i32.or
                  i32.store offset=4
                  local.get 0
                  local.get 1
                  i32.add
                  i32.const 56
                  i32.store offset=4
                  local.get 4
                  local.get 5
                  i32.const 55
                  local.get 5
                  i32.sub
                  i32.const 15
                  i32.and
                  i32.add
                  i32.const 63
                  i32.sub
                  local.tee 0
                  local.get 0
                  local.get 4
                  i32.const 16
                  i32.add
                  i32.lt_u
                  select
                  local.tee 6
                  i32.const 35
                  i32.store offset=4
                  i32.const 32480
                  i32.const 32940
                  i32.load
                  i32.store
                  i32.const 32464
                  local.get 2
                  i32.store
                  i32.const 32476
                  local.get 3
                  i32.store
                  local.get 6
                  i32.const 16
                  i32.add
                  i32.const 32908
                  i64.load align=4
                  i64.store align=4
                  local.get 6
                  i32.const 32900
                  i64.load align=4
                  i64.store offset=8 align=4
                  i32.const 32908
                  local.get 6
                  i32.const 8
                  i32.add
                  i32.store
                  i32.const 32904
                  local.get 8
                  i32.store
                  i32.const 32900
                  local.get 1
                  i32.store
                  i32.const 32912
                  i32.const 0
                  i32.store
                  local.get 6
                  i32.const 36
                  i32.add
                  local.set 0
                  loop  ;; label = @8
                    local.get 0
                    i32.const 7
                    i32.store
                    local.get 5
                    local.get 0
                    i32.const 4
                    i32.add
                    local.tee 0
                    i32.gt_u
                    br_if 0 (;@8;)
                  end
                  local.get 4
                  local.get 6
                  i32.eq
                  br_if 2 (;@5;)
                  local.get 6
                  local.get 6
                  i32.load offset=4
                  i32.const -2
                  i32.and
                  i32.store offset=4
                  local.get 6
                  local.get 6
                  local.get 4
                  i32.sub
                  local.tee 5
                  i32.store
                  local.get 4
                  local.get 5
                  i32.const 1
                  i32.or
                  i32.store offset=4
                  local.get 5
                  i32.const 255
                  i32.le_u
                  if  ;; label = @8
                    local.get 5
                    i32.const -8
                    i32.and
                    i32.const 32492
                    i32.add
                    local.set 1
                    block (result i32)  ;; label = @9
                      i32.const 32452
                      i32.load
                      local.tee 2
                      i32.const 1
                      local.get 5
                      i32.const 3
                      i32.shr_u
                      i32.shl
                      local.tee 0
                      i32.and
                      i32.eqz
                      if  ;; label = @10
                        i32.const 32452
                        local.get 0
                        local.get 2
                        i32.or
                        i32.store
                        local.get 1
                        br 1 (;@9;)
                      end
                      local.get 1
                      i32.load offset=8
                    end
                    local.tee 0
                    local.get 4
                    i32.store offset=12
                    local.get 1
                    local.get 4
                    i32.store offset=8
                    local.get 4
                    local.get 1
                    i32.store offset=12
                    local.get 4
                    local.get 0
                    i32.store offset=8
                    br 3 (;@5;)
                  end
                  i32.const 31
                  local.set 0
                  local.get 5
                  i32.const 16777215
                  i32.le_u
                  if  ;; label = @8
                    local.get 5
                    i32.const 38
                    local.get 5
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
                  local.get 4
                  local.get 0
                  i32.store offset=28
                  local.get 4
                  i64.const 0
                  i64.store offset=16 align=4
                  local.get 0
                  i32.const 2
                  i32.shl
                  i32.const 32756
                  i32.add
                  local.set 3
                  i32.const 32456
                  i32.load
                  local.tee 2
                  i32.const 1
                  local.get 0
                  i32.shl
                  local.tee 1
                  i32.and
                  i32.eqz
                  if  ;; label = @8
                    local.get 3
                    local.get 4
                    i32.store
                    i32.const 32456
                    local.get 1
                    local.get 2
                    i32.or
                    i32.store
                    local.get 4
                    local.get 3
                    i32.store offset=24
                    local.get 4
                    local.get 4
                    i32.store offset=8
                    local.get 4
                    local.get 4
                    i32.store offset=12
                    br 3 (;@5;)
                  end
                  local.get 5
                  i32.const 25
                  local.get 0
                  i32.const 1
                  i32.shr_u
                  i32.sub
                  i32.const 0
                  local.get 0
                  i32.const 31
                  i32.ne
                  select
                  i32.shl
                  local.set 0
                  local.get 3
                  i32.load
                  local.set 3
                  loop  ;; label = @8
                    local.get 3
                    local.tee 1
                    i32.load offset=4
                    i32.const -8
                    i32.and
                    local.get 5
                    i32.eq
                    br_if 2 (;@6;)
                    local.get 0
                    i32.const 29
                    i32.shr_u
                    local.set 2
                    local.get 0
                    i32.const 1
                    i32.shl
                    local.set 0
                    local.get 1
                    local.get 2
                    i32.const 4
                    i32.and
                    i32.add
                    i32.const 16
                    i32.add
                    local.tee 2
                    i32.load
                    local.tee 3
                    br_if 0 (;@8;)
                  end
                  local.get 2
                  local.get 4
                  i32.store
                  local.get 4
                  local.get 1
                  i32.store offset=24
                  local.get 4
                  local.get 4
                  i32.store offset=12
                  local.get 4
                  local.get 4
                  i32.store offset=8
                  br 2 (;@5;)
                end
                local.get 0
                local.get 1
                i32.store
                local.get 0
                local.get 0
                i32.load offset=4
                local.get 8
                i32.add
                i32.store offset=4
                local.get 1
                i32.const -8
                local.get 1
                i32.sub
                i32.const 15
                i32.and
                i32.add
                local.tee 5
                local.get 7
                i32.const 3
                i32.or
                i32.store offset=4
                local.get 2
                i32.const -8
                local.get 2
                i32.sub
                i32.const 15
                i32.and
                i32.add
                local.tee 4
                local.get 5
                local.get 7
                i32.add
                local.tee 9
                i32.sub
                local.set 6
                block  ;; label = @7
                  i32.const 32476
                  i32.load
                  local.get 4
                  i32.eq
                  if  ;; label = @8
                    i32.const 32476
                    local.get 9
                    i32.store
                    i32.const 32464
                    i32.const 32464
                    i32.load
                    local.get 6
                    i32.add
                    local.tee 0
                    i32.store
                    local.get 9
                    local.get 0
                    i32.const 1
                    i32.or
                    i32.store offset=4
                    br 1 (;@7;)
                  end
                  i32.const 32472
                  i32.load
                  local.get 4
                  i32.eq
                  if  ;; label = @8
                    i32.const 32472
                    local.get 9
                    i32.store
                    i32.const 32460
                    i32.const 32460
                    i32.load
                    local.get 6
                    i32.add
                    local.tee 0
                    i32.store
                    local.get 9
                    local.get 0
                    i32.const 1
                    i32.or
                    i32.store offset=4
                    local.get 0
                    local.get 9
                    i32.add
                    local.get 0
                    i32.store
                    br 1 (;@7;)
                  end
                  local.get 4
                  i32.load offset=4
                  local.tee 3
                  i32.const 3
                  i32.and
                  i32.const 1
                  i32.eq
                  if  ;; label = @8
                    local.get 3
                    i32.const -8
                    i32.and
                    local.set 7
                    block  ;; label = @9
                      local.get 3
                      i32.const 255
                      i32.le_u
                      if  ;; label = @10
                        local.get 3
                        i32.const 3
                        i32.shr_u
                        local.set 0
                        local.get 4
                        i32.load offset=8
                        local.tee 2
                        local.get 4
                        i32.load offset=12
                        local.tee 1
                        i32.eq
                        if  ;; label = @11
                          i32.const 32452
                          i32.const 32452
                          i32.load
                          i32.const -2
                          local.get 0
                          i32.rotl
                          i32.and
                          i32.store
                          br 2 (;@9;)
                        end
                        local.get 1
                        local.get 2
                        i32.store offset=8
                        local.get 2
                        local.get 1
                        i32.store offset=12
                        br 1 (;@9;)
                      end
                      local.get 4
                      i32.load offset=24
                      local.set 8
                      block  ;; label = @10
                        local.get 4
                        local.get 4
                        i32.load offset=12
                        local.tee 0
                        i32.ne
                        if  ;; label = @11
                          i32.const 32468
                          i32.load
                          drop
                          local.get 0
                          local.get 4
                          i32.load offset=8
                          local.tee 1
                          i32.store offset=8
                          local.get 1
                          local.get 0
                          i32.store offset=12
                          br 1 (;@10;)
                        end
                        block  ;; label = @11
                          local.get 4
                          i32.const 20
                          i32.add
                          local.tee 2
                          i32.load
                          local.tee 3
                          i32.eqz
                          if  ;; label = @12
                            local.get 4
                            i32.load offset=16
                            local.tee 3
                            i32.eqz
                            br_if 1 (;@11;)
                            local.get 4
                            i32.const 16
                            i32.add
                            local.set 2
                          end
                          loop  ;; label = @12
                            local.get 2
                            local.set 1
                            local.get 3
                            local.tee 0
                            i32.const 20
                            i32.add
                            local.tee 2
                            i32.load
                            local.tee 3
                            br_if 0 (;@12;)
                            local.get 0
                            i32.const 16
                            i32.add
                            local.set 2
                            local.get 0
                            i32.load offset=16
                            local.tee 3
                            br_if 0 (;@12;)
                          end
                          local.get 1
                          i32.const 0
                          i32.store
                          br 1 (;@10;)
                        end
                        i32.const 0
                        local.set 0
                      end
                      local.get 8
                      i32.eqz
                      br_if 0 (;@9;)
                      block  ;; label = @10
                        local.get 4
                        i32.load offset=28
                        local.tee 2
                        i32.const 2
                        i32.shl
                        i32.const 32756
                        i32.add
                        local.tee 1
                        i32.load
                        local.get 4
                        i32.eq
                        if  ;; label = @11
                          local.get 1
                          local.get 0
                          i32.store
                          local.get 0
                          br_if 1 (;@10;)
                          i32.const 32456
                          i32.const 32456
                          i32.load
                          i32.const -2
                          local.get 2
                          i32.rotl
                          i32.and
                          i32.store
                          br 2 (;@9;)
                        end
                        local.get 8
                        i32.const 16
                        i32.const 20
                        local.get 8
                        i32.load offset=16
                        local.get 4
                        i32.eq
                        select
                        i32.add
                        local.get 0
                        i32.store
                        local.get 0
                        i32.eqz
                        br_if 1 (;@9;)
                      end
                      local.get 0
                      local.get 8
                      i32.store offset=24
                      local.get 4
                      i32.load offset=16
                      local.tee 1
                      if  ;; label = @10
                        local.get 0
                        local.get 1
                        i32.store offset=16
                        local.get 1
                        local.get 0
                        i32.store offset=24
                      end
                      local.get 4
                      i32.const 20
                      i32.add
                      i32.load
                      local.tee 1
                      i32.eqz
                      br_if 0 (;@9;)
                      local.get 0
                      i32.const 20
                      i32.add
                      local.get 1
                      i32.store
                      local.get 1
                      local.get 0
                      i32.store offset=24
                    end
                    local.get 6
                    local.get 7
                    i32.add
                    local.set 6
                    local.get 4
                    local.get 7
                    i32.add
                    local.tee 4
                    i32.load offset=4
                    local.set 3
                  end
                  local.get 4
                  local.get 3
                  i32.const -2
                  i32.and
                  i32.store offset=4
                  local.get 6
                  local.get 9
                  i32.add
                  local.get 6
                  i32.store
                  local.get 9
                  local.get 6
                  i32.const 1
                  i32.or
                  i32.store offset=4
                  local.get 6
                  i32.const 255
                  i32.le_u
                  if  ;; label = @8
                    local.get 6
                    i32.const -8
                    i32.and
                    i32.const 32492
                    i32.add
                    local.set 1
                    block (result i32)  ;; label = @9
                      i32.const 32452
                      i32.load
                      local.tee 2
                      i32.const 1
                      local.get 6
                      i32.const 3
                      i32.shr_u
                      i32.shl
                      local.tee 0
                      i32.and
                      i32.eqz
                      if  ;; label = @10
                        i32.const 32452
                        local.get 0
                        local.get 2
                        i32.or
                        i32.store
                        local.get 1
                        br 1 (;@9;)
                      end
                      local.get 1
                      i32.load offset=8
                    end
                    local.tee 0
                    local.get 9
                    i32.store offset=12
                    local.get 1
                    local.get 9
                    i32.store offset=8
                    local.get 9
                    local.get 1
                    i32.store offset=12
                    local.get 9
                    local.get 0
                    i32.store offset=8
                    br 1 (;@7;)
                  end
                  i32.const 31
                  local.set 3
                  local.get 6
                  i32.const 16777215
                  i32.le_u
                  if  ;; label = @8
                    local.get 6
                    i32.const 38
                    local.get 6
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
                    local.set 3
                  end
                  local.get 9
                  local.get 3
                  i32.store offset=28
                  local.get 9
                  i64.const 0
                  i64.store offset=16 align=4
                  local.get 3
                  i32.const 2
                  i32.shl
                  i32.const 32756
                  i32.add
                  local.set 2
                  i32.const 32456
                  i32.load
                  local.tee 1
                  i32.const 1
                  local.get 3
                  i32.shl
                  local.tee 0
                  i32.and
                  i32.eqz
                  if  ;; label = @8
                    local.get 2
                    local.get 9
                    i32.store
                    i32.const 32456
                    local.get 0
                    local.get 1
                    i32.or
                    i32.store
                    local.get 9
                    local.get 2
                    i32.store offset=24
                    local.get 9
                    local.get 9
                    i32.store offset=8
                    local.get 9
                    local.get 9
                    i32.store offset=12
                    br 1 (;@7;)
                  end
                  local.get 6
                  i32.const 25
                  local.get 3
                  i32.const 1
                  i32.shr_u
                  i32.sub
                  i32.const 0
                  local.get 3
                  i32.const 31
                  i32.ne
                  select
                  i32.shl
                  local.set 3
                  local.get 2
                  i32.load
                  local.set 0
                  block  ;; label = @8
                    loop  ;; label = @9
                      local.get 0
                      local.tee 1
                      i32.load offset=4
                      i32.const -8
                      i32.and
                      local.get 6
                      i32.eq
                      br_if 1 (;@8;)
                      local.get 3
                      i32.const 29
                      i32.shr_u
                      local.set 0
                      local.get 3
                      i32.const 1
                      i32.shl
                      local.set 3
                      local.get 1
                      local.get 0
                      i32.const 4
                      i32.and
                      i32.add
                      i32.const 16
                      i32.add
                      local.tee 2
                      i32.load
                      local.tee 0
                      br_if 0 (;@9;)
                    end
                    local.get 2
                    local.get 9
                    i32.store
                    local.get 9
                    local.get 1
                    i32.store offset=24
                    local.get 9
                    local.get 9
                    i32.store offset=12
                    local.get 9
                    local.get 9
                    i32.store offset=8
                    br 1 (;@7;)
                  end
                  local.get 1
                  i32.load offset=8
                  local.tee 0
                  local.get 9
                  i32.store offset=12
                  local.get 1
                  local.get 9
                  i32.store offset=8
                  local.get 9
                  i32.const 0
                  i32.store offset=24
                  local.get 9
                  local.get 1
                  i32.store offset=12
                  local.get 9
                  local.get 0
                  i32.store offset=8
                end
                local.get 5
                i32.const 8
                i32.add
                local.set 0
                br 5 (;@1;)
              end
              local.get 1
              i32.load offset=8
              local.tee 0
              local.get 4
              i32.store offset=12
              local.get 1
              local.get 4
              i32.store offset=8
              local.get 4
              i32.const 0
              i32.store offset=24
              local.get 4
              local.get 1
              i32.store offset=12
              local.get 4
              local.get 0
              i32.store offset=8
            end
            i32.const 32464
            i32.load
            local.tee 0
            local.get 7
            i32.le_u
            br_if 0 (;@4;)
            i32.const 32476
            i32.load
            local.tee 2
            local.get 7
            i32.add
            local.tee 1
            local.get 0
            local.get 7
            i32.sub
            local.tee 0
            i32.const 1
            i32.or
            i32.store offset=4
            i32.const 32464
            local.get 0
            i32.store
            i32.const 32476
            local.get 1
            i32.store
            local.get 2
            local.get 7
            i32.const 3
            i32.or
            i32.store offset=4
            local.get 2
            i32.const 8
            i32.add
            local.set 0
            br 3 (;@1;)
          end
          i32.const 0
          local.set 0
          i32.const 32448
          i32.const 48
          i32.store
          br 2 (;@1;)
        end
        block  ;; label = @3
          local.get 8
          i32.eqz
          br_if 0 (;@3;)
          block  ;; label = @4
            local.get 3
            i32.load offset=28
            local.tee 2
            i32.const 2
            i32.shl
            i32.const 32756
            i32.add
            local.tee 0
            i32.load
            local.get 3
            i32.eq
            if  ;; label = @5
              local.get 0
              local.get 1
              i32.store
              local.get 1
              br_if 1 (;@4;)
              i32.const 32456
              local.get 4
              i32.const -2
              local.get 2
              i32.rotl
              i32.and
              local.tee 4
              i32.store
              br 2 (;@3;)
            end
            local.get 8
            i32.const 16
            i32.const 20
            local.get 8
            i32.load offset=16
            local.get 3
            i32.eq
            select
            i32.add
            local.get 1
            i32.store
            local.get 1
            i32.eqz
            br_if 1 (;@3;)
          end
          local.get 1
          local.get 8
          i32.store offset=24
          local.get 3
          i32.load offset=16
          local.tee 0
          if  ;; label = @4
            local.get 1
            local.get 0
            i32.store offset=16
            local.get 0
            local.get 1
            i32.store offset=24
          end
          local.get 3
          i32.const 20
          i32.add
          i32.load
          local.tee 0
          i32.eqz
          br_if 0 (;@3;)
          local.get 1
          i32.const 20
          i32.add
          local.get 0
          i32.store
          local.get 0
          local.get 1
          i32.store offset=24
        end
        block  ;; label = @3
          local.get 6
          i32.const 15
          i32.le_u
          if  ;; label = @4
            local.get 3
            local.get 6
            local.get 7
            i32.or
            local.tee 0
            i32.const 3
            i32.or
            i32.store offset=4
            local.get 0
            local.get 3
            i32.add
            local.tee 0
            local.get 0
            i32.load offset=4
            i32.const 1
            i32.or
            i32.store offset=4
            br 1 (;@3;)
          end
          local.get 3
          local.get 7
          i32.add
          local.tee 5
          local.get 6
          i32.const 1
          i32.or
          i32.store offset=4
          local.get 3
          local.get 7
          i32.const 3
          i32.or
          i32.store offset=4
          local.get 5
          local.get 6
          i32.add
          local.get 6
          i32.store
          local.get 6
          i32.const 255
          i32.le_u
          if  ;; label = @4
            local.get 6
            i32.const -8
            i32.and
            i32.const 32492
            i32.add
            local.set 1
            block (result i32)  ;; label = @5
              i32.const 32452
              i32.load
              local.tee 2
              i32.const 1
              local.get 6
              i32.const 3
              i32.shr_u
              i32.shl
              local.tee 0
              i32.and
              i32.eqz
              if  ;; label = @6
                i32.const 32452
                local.get 0
                local.get 2
                i32.or
                i32.store
                local.get 1
                br 1 (;@5;)
              end
              local.get 1
              i32.load offset=8
            end
            local.tee 0
            local.get 5
            i32.store offset=12
            local.get 1
            local.get 5
            i32.store offset=8
            local.get 5
            local.get 1
            i32.store offset=12
            local.get 5
            local.get 0
            i32.store offset=8
            br 1 (;@3;)
          end
          i32.const 31
          local.set 0
          local.get 6
          i32.const 16777215
          i32.le_u
          if  ;; label = @4
            local.get 6
            i32.const 38
            local.get 6
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
          i32.const 32756
          i32.add
          local.set 2
          i32.const 1
          local.get 0
          i32.shl
          local.tee 1
          local.get 4
          i32.and
          i32.eqz
          if  ;; label = @4
            local.get 2
            local.get 5
            i32.store
            i32.const 32456
            local.get 1
            local.get 4
            i32.or
            i32.store
            local.get 5
            local.get 2
            i32.store offset=24
            local.get 5
            local.get 5
            i32.store offset=8
            local.get 5
            local.get 5
            i32.store offset=12
            br 1 (;@3;)
          end
          local.get 6
          i32.const 25
          local.get 0
          i32.const 1
          i32.shr_u
          i32.sub
          i32.const 0
          local.get 0
          i32.const 31
          i32.ne
          select
          i32.shl
          local.set 0
          local.get 2
          i32.load
          local.set 7
          block  ;; label = @4
            loop  ;; label = @5
              local.get 7
              local.tee 1
              i32.load offset=4
              i32.const -8
              i32.and
              local.get 6
              i32.eq
              br_if 1 (;@4;)
              local.get 0
              i32.const 29
              i32.shr_u
              local.set 2
              local.get 0
              i32.const 1
              i32.shl
              local.set 0
              local.get 1
              local.get 2
              i32.const 4
              i32.and
              i32.add
              i32.const 16
              i32.add
              local.tee 2
              i32.load
              local.tee 7
              br_if 0 (;@5;)
            end
            local.get 2
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
            br 1 (;@3;)
          end
          local.get 1
          i32.load offset=8
          local.tee 0
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
          local.get 0
          i32.store offset=8
        end
        local.get 3
        i32.const 8
        i32.add
        local.set 0
        br 1 (;@1;)
      end
      block  ;; label = @2
        local.get 9
        i32.eqz
        br_if 0 (;@2;)
        block  ;; label = @3
          local.get 1
          i32.load offset=28
          local.tee 2
          i32.const 2
          i32.shl
          i32.const 32756
          i32.add
          local.tee 0
          i32.load
          local.get 1
          i32.eq
          if  ;; label = @4
            local.get 0
            local.get 3
            i32.store
            local.get 3
            br_if 1 (;@3;)
            i32.const 32456
            local.get 4
            i32.const -2
            local.get 2
            i32.rotl
            i32.and
            i32.store
            br 2 (;@2;)
          end
          local.get 9
          i32.const 16
          i32.const 20
          local.get 9
          i32.load offset=16
          local.get 1
          i32.eq
          select
          i32.add
          local.get 3
          i32.store
          local.get 3
          i32.eqz
          br_if 1 (;@2;)
        end
        local.get 3
        local.get 9
        i32.store offset=24
        local.get 1
        i32.load offset=16
        local.tee 0
        if  ;; label = @3
          local.get 3
          local.get 0
          i32.store offset=16
          local.get 0
          local.get 3
          i32.store offset=24
        end
        local.get 1
        i32.const 20
        i32.add
        i32.load
        local.tee 0
        i32.eqz
        br_if 0 (;@2;)
        local.get 3
        i32.const 20
        i32.add
        local.get 0
        i32.store
        local.get 0
        local.get 3
        i32.store offset=24
      end
      block  ;; label = @2
        local.get 6
        i32.const 15
        i32.le_u
        if  ;; label = @3
          local.get 1
          local.get 6
          local.get 7
          i32.or
          local.tee 0
          i32.const 3
          i32.or
          i32.store offset=4
          local.get 0
          local.get 1
          i32.add
          local.tee 0
          local.get 0
          i32.load offset=4
          i32.const 1
          i32.or
          i32.store offset=4
          br 1 (;@2;)
        end
        local.get 1
        local.get 7
        i32.add
        local.tee 3
        local.get 6
        i32.const 1
        i32.or
        i32.store offset=4
        local.get 1
        local.get 7
        i32.const 3
        i32.or
        i32.store offset=4
        local.get 3
        local.get 6
        i32.add
        local.get 6
        i32.store
        local.get 10
        if  ;; label = @3
          local.get 10
          i32.const -8
          i32.and
          i32.const 32492
          i32.add
          local.set 2
          i32.const 32472
          i32.load
          local.set 7
          block (result i32)  ;; label = @4
            i32.const 1
            local.get 10
            i32.const 3
            i32.shr_u
            i32.shl
            local.tee 0
            local.get 8
            i32.and
            i32.eqz
            if  ;; label = @5
              i32.const 32452
              local.get 0
              local.get 8
              i32.or
              i32.store
              local.get 2
              br 1 (;@4;)
            end
            local.get 2
            i32.load offset=8
          end
          local.tee 0
          local.get 7
          i32.store offset=12
          local.get 2
          local.get 7
          i32.store offset=8
          local.get 7
          local.get 2
          i32.store offset=12
          local.get 7
          local.get 0
          i32.store offset=8
        end
        i32.const 32472
        local.get 3
        i32.store
        i32.const 32460
        local.get 6
        i32.store
      end
      local.get 1
      i32.const 8
      i32.add
      local.set 0
    end
    local.get 11
    i32.const 16
    i32.add
    global.set 0
    local.get 0)
  (func $free (type 1) (param i32)
    (local i32 i32 i32 i32 i32 i32 i32)
    block  ;; label = @1
      local.get 0
      i32.eqz
      br_if 0 (;@1;)
      local.get 0
      i32.const 4
      i32.sub
      i32.load
      local.tee 2
      i32.const -8
      i32.and
      local.tee 4
      local.get 0
      i32.const 8
      i32.sub
      local.tee 1
      i32.add
      local.set 5
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
        local.tee 0
        i32.sub
        local.tee 1
        i32.const 32468
        i32.load
        i32.lt_u
        br_if 1 (;@1;)
        local.get 0
        local.get 4
        i32.add
        local.set 4
        block  ;; label = @3
          block  ;; label = @4
            i32.const 32472
            i32.load
            local.get 1
            i32.ne
            if  ;; label = @5
              local.get 0
              i32.const 255
              i32.le_u
              if  ;; label = @6
                local.get 0
                i32.const 3
                i32.shr_u
                local.set 0
                local.get 1
                i32.load offset=8
                local.tee 3
                local.get 1
                i32.load offset=12
                local.tee 2
                i32.eq
                if  ;; label = @7
                  i32.const 32452
                  i32.const 32452
                  i32.load
                  i32.const -2
                  local.get 0
                  i32.rotl
                  i32.and
                  i32.store
                  br 5 (;@2;)
                end
                local.get 2
                local.get 3
                i32.store offset=8
                local.get 3
                local.get 2
                i32.store offset=12
                br 4 (;@2;)
              end
              local.get 1
              i32.load offset=24
              local.set 6
              local.get 1
              local.get 1
              i32.load offset=12
              local.tee 0
              i32.ne
              if  ;; label = @6
                local.get 0
                local.get 1
                i32.load offset=8
                local.tee 2
                i32.store offset=8
                local.get 2
                local.get 0
                i32.store offset=12
                br 3 (;@3;)
              end
              local.get 1
              i32.const 20
              i32.add
              local.tee 2
              i32.load
              local.tee 3
              i32.eqz
              if  ;; label = @6
                local.get 1
                i32.load offset=16
                local.tee 3
                i32.eqz
                br_if 2 (;@4;)
                local.get 1
                i32.const 16
                i32.add
                local.set 2
              end
              loop  ;; label = @6
                local.get 2
                local.set 7
                local.get 3
                local.tee 0
                i32.const 20
                i32.add
                local.tee 2
                i32.load
                local.tee 3
                br_if 0 (;@6;)
                local.get 0
                i32.const 16
                i32.add
                local.set 2
                local.get 0
                i32.load offset=16
                local.tee 3
                br_if 0 (;@6;)
              end
              local.get 7
              i32.const 0
              i32.store
              br 2 (;@3;)
            end
            local.get 5
            i32.load offset=4
            local.tee 0
            i32.const 3
            i32.and
            i32.const 3
            i32.ne
            br_if 2 (;@2;)
            local.get 5
            local.get 0
            i32.const -2
            i32.and
            i32.store offset=4
            i32.const 32460
            local.get 4
            i32.store
            local.get 5
            local.get 4
            i32.store
            local.get 1
            local.get 4
            i32.const 1
            i32.or
            i32.store offset=4
            br 3 (;@1;)
          end
          i32.const 0
          local.set 0
        end
        local.get 6
        i32.eqz
        br_if 0 (;@2;)
        block  ;; label = @3
          local.get 1
          i32.load offset=28
          local.tee 3
          i32.const 2
          i32.shl
          i32.const 32756
          i32.add
          local.tee 2
          i32.load
          local.get 1
          i32.eq
          if  ;; label = @4
            local.get 2
            local.get 0
            i32.store
            local.get 0
            br_if 1 (;@3;)
            i32.const 32456
            i32.const 32456
            i32.load
            i32.const -2
            local.get 3
            i32.rotl
            i32.and
            i32.store
            br 2 (;@2;)
          end
          local.get 6
          i32.const 16
          i32.const 20
          local.get 6
          i32.load offset=16
          local.get 1
          i32.eq
          select
          i32.add
          local.get 0
          i32.store
          local.get 0
          i32.eqz
          br_if 1 (;@2;)
        end
        local.get 0
        local.get 6
        i32.store offset=24
        local.get 1
        i32.load offset=16
        local.tee 2
        if  ;; label = @3
          local.get 0
          local.get 2
          i32.store offset=16
          local.get 2
          local.get 0
          i32.store offset=24
        end
        local.get 1
        i32.const 20
        i32.add
        i32.load
        local.tee 2
        i32.eqz
        br_if 0 (;@2;)
        local.get 0
        i32.const 20
        i32.add
        local.get 2
        i32.store
        local.get 2
        local.get 0
        i32.store offset=24
      end
      local.get 1
      local.get 5
      i32.ge_u
      br_if 0 (;@1;)
      local.get 5
      i32.load offset=4
      local.tee 0
      i32.const 1
      i32.and
      i32.eqz
      br_if 0 (;@1;)
      block  ;; label = @2
        block  ;; label = @3
          block  ;; label = @4
            block  ;; label = @5
              local.get 0
              i32.const 2
              i32.and
              i32.eqz
              if  ;; label = @6
                i32.const 32476
                i32.load
                local.get 5
                i32.eq
                if  ;; label = @7
                  i32.const 32476
                  local.get 1
                  i32.store
                  i32.const 32464
                  i32.const 32464
                  i32.load
                  local.get 4
                  i32.add
                  local.tee 0
                  i32.store
                  local.get 1
                  local.get 0
                  i32.const 1
                  i32.or
                  i32.store offset=4
                  local.get 1
                  i32.const 32472
                  i32.load
                  i32.ne
                  br_if 6 (;@1;)
                  i32.const 32460
                  i32.const 0
                  i32.store
                  i32.const 32472
                  i32.const 0
                  i32.store
                  br 6 (;@1;)
                end
                i32.const 32472
                i32.load
                local.get 5
                i32.eq
                if  ;; label = @7
                  i32.const 32472
                  local.get 1
                  i32.store
                  i32.const 32460
                  i32.const 32460
                  i32.load
                  local.get 4
                  i32.add
                  local.tee 0
                  i32.store
                  local.get 1
                  local.get 0
                  i32.const 1
                  i32.or
                  i32.store offset=4
                  local.get 0
                  local.get 1
                  i32.add
                  local.get 0
                  i32.store
                  br 6 (;@1;)
                end
                local.get 0
                i32.const -8
                i32.and
                local.get 4
                i32.add
                local.set 4
                local.get 0
                i32.const 255
                i32.le_u
                if  ;; label = @7
                  local.get 0
                  i32.const 3
                  i32.shr_u
                  local.set 0
                  local.get 5
                  i32.load offset=8
                  local.tee 3
                  local.get 5
                  i32.load offset=12
                  local.tee 2
                  i32.eq
                  if  ;; label = @8
                    i32.const 32452
                    i32.const 32452
                    i32.load
                    i32.const -2
                    local.get 0
                    i32.rotl
                    i32.and
                    i32.store
                    br 5 (;@3;)
                  end
                  local.get 2
                  local.get 3
                  i32.store offset=8
                  local.get 3
                  local.get 2
                  i32.store offset=12
                  br 4 (;@3;)
                end
                local.get 5
                i32.load offset=24
                local.set 6
                local.get 5
                local.get 5
                i32.load offset=12
                local.tee 0
                i32.ne
                if  ;; label = @7
                  i32.const 32468
                  i32.load
                  drop
                  local.get 0
                  local.get 5
                  i32.load offset=8
                  local.tee 2
                  i32.store offset=8
                  local.get 2
                  local.get 0
                  i32.store offset=12
                  br 3 (;@4;)
                end
                local.get 5
                i32.const 20
                i32.add
                local.tee 2
                i32.load
                local.tee 3
                i32.eqz
                if  ;; label = @7
                  local.get 5
                  i32.load offset=16
                  local.tee 3
                  i32.eqz
                  br_if 2 (;@5;)
                  local.get 5
                  i32.const 16
                  i32.add
                  local.set 2
                end
                loop  ;; label = @7
                  local.get 2
                  local.set 7
                  local.get 3
                  local.tee 0
                  i32.const 20
                  i32.add
                  local.tee 2
                  i32.load
                  local.tee 3
                  br_if 0 (;@7;)
                  local.get 0
                  i32.const 16
                  i32.add
                  local.set 2
                  local.get 0
                  i32.load offset=16
                  local.tee 3
                  br_if 0 (;@7;)
                end
                local.get 7
                i32.const 0
                i32.store
                br 2 (;@4;)
              end
              local.get 5
              local.get 0
              i32.const -2
              i32.and
              i32.store offset=4
              local.get 1
              local.get 4
              i32.add
              local.get 4
              i32.store
              local.get 1
              local.get 4
              i32.const 1
              i32.or
              i32.store offset=4
              br 3 (;@2;)
            end
            i32.const 0
            local.set 0
          end
          local.get 6
          i32.eqz
          br_if 0 (;@3;)
          block  ;; label = @4
            local.get 5
            i32.load offset=28
            local.tee 3
            i32.const 2
            i32.shl
            i32.const 32756
            i32.add
            local.tee 2
            i32.load
            local.get 5
            i32.eq
            if  ;; label = @5
              local.get 2
              local.get 0
              i32.store
              local.get 0
              br_if 1 (;@4;)
              i32.const 32456
              i32.const 32456
              i32.load
              i32.const -2
              local.get 3
              i32.rotl
              i32.and
              i32.store
              br 2 (;@3;)
            end
            local.get 6
            i32.const 16
            i32.const 20
            local.get 6
            i32.load offset=16
            local.get 5
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
          local.get 6
          i32.store offset=24
          local.get 5
          i32.load offset=16
          local.tee 2
          if  ;; label = @4
            local.get 0
            local.get 2
            i32.store offset=16
            local.get 2
            local.get 0
            i32.store offset=24
          end
          local.get 5
          i32.const 20
          i32.add
          i32.load
          local.tee 2
          i32.eqz
          br_if 0 (;@3;)
          local.get 0
          i32.const 20
          i32.add
          local.get 2
          i32.store
          local.get 2
          local.get 0
          i32.store offset=24
        end
        local.get 1
        local.get 4
        i32.add
        local.get 4
        i32.store
        local.get 1
        local.get 4
        i32.const 1
        i32.or
        i32.store offset=4
        local.get 1
        i32.const 32472
        i32.load
        i32.ne
        br_if 0 (;@2;)
        i32.const 32460
        local.get 4
        i32.store
        br 1 (;@1;)
      end
      local.get 4
      i32.const 255
      i32.le_u
      if  ;; label = @2
        local.get 4
        i32.const -8
        i32.and
        i32.const 32492
        i32.add
        local.set 2
        block (result i32)  ;; label = @3
          i32.const 32452
          i32.load
          local.tee 3
          i32.const 1
          local.get 4
          i32.const 3
          i32.shr_u
          i32.shl
          local.tee 0
          i32.and
          i32.eqz
          if  ;; label = @4
            i32.const 32452
            local.get 0
            local.get 3
            i32.or
            i32.store
            local.get 2
            br 1 (;@3;)
          end
          local.get 2
          i32.load offset=8
        end
        local.tee 0
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
        br 1 (;@1;)
      end
      i32.const 31
      local.set 3
      local.get 4
      i32.const 16777215
      i32.le_u
      if  ;; label = @2
        local.get 4
        i32.const 38
        local.get 4
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
        local.set 3
      end
      local.get 1
      local.get 3
      i32.store offset=28
      local.get 1
      i64.const 0
      i64.store offset=16 align=4
      local.get 3
      i32.const 2
      i32.shl
      i32.const 32756
      i32.add
      local.set 7
      block  ;; label = @2
        i32.const 32456
        i32.load
        local.tee 2
        i32.const 1
        local.get 3
        i32.shl
        local.tee 0
        i32.and
        i32.eqz
        if  ;; label = @3
          local.get 7
          local.get 1
          i32.store
          i32.const 32456
          local.get 0
          local.get 2
          i32.or
          i32.store
          local.get 1
          local.get 7
          i32.store offset=24
          local.get 1
          local.get 1
          i32.store offset=8
          local.get 1
          local.get 1
          i32.store offset=12
          br 1 (;@2;)
        end
        local.get 4
        i32.const 25
        local.get 3
        i32.const 1
        i32.shr_u
        i32.sub
        i32.const 0
        local.get 3
        i32.const 31
        i32.ne
        select
        i32.shl
        local.set 3
        local.get 7
        i32.load
        local.set 0
        block  ;; label = @3
          loop  ;; label = @4
            local.get 0
            local.tee 2
            i32.load offset=4
            i32.const -8
            i32.and
            local.get 4
            i32.eq
            br_if 1 (;@3;)
            local.get 3
            i32.const 29
            i32.shr_u
            local.set 0
            local.get 3
            i32.const 1
            i32.shl
            local.set 3
            local.get 2
            local.get 0
            i32.const 4
            i32.and
            i32.add
            i32.const 16
            i32.add
            local.tee 7
            i32.load
            local.tee 0
            br_if 0 (;@4;)
          end
          local.get 7
          local.get 1
          i32.store
          local.get 1
          local.get 2
          i32.store offset=24
          local.get 1
          local.get 1
          i32.store offset=12
          local.get 1
          local.get 1
          i32.store offset=8
          br 1 (;@2;)
        end
        local.get 2
        i32.load offset=8
        local.tee 0
        local.get 1
        i32.store offset=12
        local.get 2
        local.get 1
        i32.store offset=8
        local.get 1
        i32.const 0
        i32.store offset=24
        local.get 1
        local.get 2
        i32.store offset=12
        local.get 1
        local.get 0
        i32.store offset=8
      end
      i32.const 32484
      i32.const 32484
      i32.load
      i32.const 1
      i32.sub
      local.tee 0
      i32.const -1
      local.get 0
      select
      i32.store
    end)