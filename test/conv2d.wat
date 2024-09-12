(module
  (type (;0;) (func (param i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)))
  (type (;1;) (func (param i32) (result i32)))
  (import "env" "__linear_memory" (memory (;0;) 1))
  (import "env" "malloc" (func (;0;) (type 1)))
  (func $main (type 0) (param i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
    (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 f32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 f32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 f32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 f32 f32 f32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
    local.get 9
    local.set 10
    local.get 9
    local.set 11
    local.get 8
    local.set 12
    local.get 7
    local.set 13
    i32.const 1
    local.set 14
    local.get 14
    drop
    i32.const 28
    local.set 15
    local.get 15
    drop
    local.get 3
    local.set 16
    local.get 2
    local.set 17
    local.get 1
    drop
    i32.const 32512
    local.set 18
    local.get 18
    call 0
    local.set 19
    i32.const 63
    local.set 20
    local.get 19
    local.get 20
    i32.add
    local.set 21
    i32.const -64
    local.set 22
    local.get 21
    local.get 22
    i32.and
    local.set 23
    local.get 14
    local.set 24
    i32.const 12
    local.set 25
    local.get 25
    local.set 26
    i32.const 312
    local.set 27
    i32.const 8112
    local.set 28
    i32.const 26
    local.set 29
    local.get 29
    local.set 30
    local.get 14
    local.set 31
    i32.const 0
    local.set 32
    i32.const 0
    local.set 33
    local.get 33
    local.set 34
    block  ;; label = @1
      loop  ;; label = @2
        local.get 34
        local.set 35
        i32.const 1
        local.set 36
        local.get 35
        local.get 36
        i32.lt_s
        local.set 37
        i32.const 1
        local.set 38
        local.get 37
        local.get 38
        i32.and
        local.set 39
        local.get 39
        i32.eqz
        br_if 1 (;@1;)
        i32.const 0
        local.set 40
        local.get 40
        local.set 41
        block  ;; label = @3
          loop  ;; label = @4
            local.get 41
            local.set 42
            i32.const 26
            local.set 43
            local.get 42
            local.get 43
            i32.lt_s
            local.set 44
            i32.const 1
            local.set 45
            local.get 44
            local.get 45
            i32.and
            local.set 46
            local.get 46
            i32.eqz
            br_if 1 (;@3;)
            i32.const 0
            local.set 47
            local.get 47
            local.set 48
            block  ;; label = @5
              loop  ;; label = @6
                local.get 48
                local.set 49
                i32.const 26
                local.set 50
                local.get 49
                local.get 50
                i32.lt_s
                local.set 51
                i32.const 1
                local.set 52
                local.get 51
                local.get 52
                i32.and
                local.set 53
                local.get 53
                i32.eqz
                br_if 1 (;@5;)
                i32.const 0
                local.set 54
                local.get 54
                local.set 55
                block  ;; label = @7
                  loop  ;; label = @8
                    local.get 55
                    local.set 56
                    i32.const 12
                    local.set 57
                    local.get 56
                    local.get 57
                    i32.lt_s
                    local.set 58
                    i32.const 1
                    local.set 59
                    local.get 58
                    local.get 59
                    i32.and
                    local.set 60
                    local.get 60
                    i32.eqz
                    br_if 1 (;@7;)
                    i32.const 0
                    local.set 61
                    i32.const 2
                    local.set 62
                    local.get 56
                    local.get 62
                    i32.shl
                    local.set 63
                    local.get 61
                    local.get 63
                    i32.add
                    local.set 64
                    local.get 64
                    f32.load
                    local.set 65
                    i32.const 8112
                    local.set 66
                    local.get 35
                    local.get 66
                    i32.mul
                    local.set 67
                    i32.const 312
                    local.set 68
                    local.get 42
                    local.get 68
                    i32.mul
                    local.set 69
                    local.get 67
                    local.get 69
                    i32.add
                    local.set 70
                    i32.const 12
                    local.set 71
                    local.get 49
                    local.get 71
                    i32.mul
                    local.set 72
                    local.get 70
                    local.get 72
                    i32.add
                    local.set 73
                    local.get 73
                    local.get 56
                    i32.add
                    local.set 74
                    i32.const 2
                    local.set 75
                    local.get 74
                    local.get 75
                    i32.shl
                    local.set 76
                    local.get 23
                    local.get 76
                    i32.add
                    local.set 77
                    local.get 77
                    local.get 65
                    f32.store
                    i32.const 1
                    local.set 78
                    local.get 56
                    local.get 78
                    i32.add
                    local.set 79
                    local.get 79
                    local.set 55
                    br 0 (;@8;)
                  end
                end
                i32.const 1
                local.set 80
                local.get 49
                local.get 80
                i32.add
                local.set 81
                local.get 81
                local.set 48
                br 0 (;@6;)
              end
            end
            i32.const 1
            local.set 82
            local.get 42
            local.get 82
            i32.add
            local.set 83
            local.get 83
            local.set 41
            br 0 (;@4;)
          end
        end
        i32.const 1
        local.set 84
        local.get 35
        local.get 84
        i32.add
        local.set 85
        local.get 85
        local.set 34
        br 0 (;@2;)
      end
    end
    i32.const 0
    local.set 86
    local.get 86
    local.set 87
    block  ;; label = @1
      loop  ;; label = @2
        local.get 87
        local.set 88
        i32.const 1
        local.set 89
        local.get 88
        local.get 89
        i32.lt_s
        local.set 90
        i32.const 1
        local.set 91
        local.get 90
        local.get 91
        i32.and
        local.set 92
        local.get 92
        i32.eqz
        br_if 1 (;@1;)
        i32.const 0
        local.set 93
        local.get 93
        local.set 94
        block  ;; label = @3
          loop  ;; label = @4
            local.get 94
            local.set 95
            i32.const 26
            local.set 96
            local.get 95
            local.get 96
            i32.lt_s
            local.set 97
            i32.const 1
            local.set 98
            local.get 97
            local.get 98
            i32.and
            local.set 99
            local.get 99
            i32.eqz
            br_if 1 (;@3;)
            i32.const 0
            local.set 100
            local.get 100
            local.set 101
            block  ;; label = @5
              loop  ;; label = @6
                local.get 101
                local.set 102
                i32.const 26
                local.set 103
                local.get 102
                local.get 103
                i32.lt_s
                local.set 104
                i32.const 1
                local.set 105
                local.get 104
                local.get 105
                i32.and
                local.set 106
                local.get 106
                i32.eqz
                br_if 1 (;@5;)
                i32.const 0
                local.set 107
                local.get 107
                local.set 108
                block  ;; label = @7
                  loop  ;; label = @8
                    local.get 108
                    local.set 109
                    i32.const 12
                    local.set 110
                    local.get 109
                    local.get 110
                    i32.lt_s
                    local.set 111
                    i32.const 1
                    local.set 112
                    local.get 111
                    local.get 112
                    i32.and
                    local.set 113
                    local.get 113
                    i32.eqz
                    br_if 1 (;@7;)
                    i32.const 0
                    local.set 114
                    local.get 114
                    local.set 115
                    block  ;; label = @9
                      loop  ;; label = @10
                        local.get 115
                        local.set 116
                        i32.const 3
                        local.set 117
                        local.get 116
                        local.get 117
                        i32.lt_s
                        local.set 118
                        i32.const 1
                        local.set 119
                        local.get 118
                        local.get 119
                        i32.and
                        local.set 120
                        local.get 120
                        i32.eqz
                        br_if 1 (;@9;)
                        i32.const 0
                        local.set 121
                        local.get 121
                        local.set 122
                        block  ;; label = @11
                          loop  ;; label = @12
                            local.get 122
                            local.set 123
                            i32.const 3
                            local.set 124
                            local.get 123
                            local.get 124
                            i32.lt_s
                            local.set 125
                            i32.const 1
                            local.set 126
                            local.get 125
                            local.get 126
                            i32.and
                            local.set 127
                            local.get 127
                            i32.eqz
                            br_if 1 (;@11;)
                            i32.const 0
                            local.set 128
                            local.get 128
                            local.set 129
                            block  ;; label = @13
                              loop  ;; label = @14
                                local.get 129
                                local.set 130
                                i32.const 1
                                local.set 131
                                local.get 130
                                local.get 131
                                i32.lt_s
                                local.set 132
                                i32.const 1
                                local.set 133
                                local.get 132
                                local.get 133
                                i32.and
                                local.set 134
                                local.get 134
                                i32.eqz
                                br_if 1 (;@13;)
                                local.get 95
                                local.get 116
                                i32.add
                                local.set 135
                                local.get 102
                                local.get 123
                                i32.add
                                local.set 136
                                i32.const 2
                                local.set 137
                                local.get 16
                                local.get 137
                                i32.shl
                                local.set 138
                                local.get 17
                                local.get 138
                                i32.add
                                local.set 139
                                local.get 88
                                local.get 13
                                i32.mul
                                local.set 140
                                local.get 135
                                local.get 12
                                i32.mul
                                local.set 141
                                local.get 140
                                local.get 141
                                i32.add
                                local.set 142
                                local.get 136
                                local.get 11
                                i32.mul
                                local.set 143
                                local.get 142
                                local.get 143
                                i32.add
                                local.set 144
                                local.get 130
                                local.get 10
                                i32.mul
                                local.set 145
                                local.get 144
                                local.get 145
                                i32.add
                                local.set 146
                                i32.const 2
                                local.set 147
                                local.get 146
                                local.get 147
                                i32.shl
                                local.set 148
                                local.get 139
                                local.get 148
                                i32.add
                                local.set 149
                                local.get 149
                                f32.load
                                local.set 150
                                i32.const 9
                                local.set 151
                                local.get 109
                                local.get 151
                                i32.mul
                                local.set 152
                                i32.const 3
                                local.set 153
                                local.get 116
                                local.get 153
                                i32.mul
                                local.set 154
                                local.get 152
                                local.get 154
                                i32.add
                                local.set 155
                                local.get 155
                                local.get 123
                                i32.add
                                local.set 156
                                local.get 156
                                local.get 130
                                i32.add
                                local.set 157
                                i32.const 64
                                local.set 158
                                i32.const 2
                                local.set 159
                                local.get 157
                                local.get 159
                                i32.shl
                                local.set 160
                                local.get 158
                                local.get 160
                                i32.add
                                local.set 161
                                local.get 161
                                f32.load
                                local.set 162
                                i32.const 8112
                                local.set 163
                                local.get 88
                                local.get 163
                                i32.mul
                                local.set 164
                                i32.const 312
                                local.set 165
                                local.get 95
                                local.get 165
                                i32.mul
                                local.set 166
                                local.get 164
                                local.get 166
                                i32.add
                                local.set 167
                                i32.const 12
                                local.set 168
                                local.get 102
                                local.get 168
                                i32.mul
                                local.set 169
                                local.get 167
                                local.get 169
                                i32.add
                                local.set 170
                                local.get 170
                                local.get 109
                                i32.add
                                local.set 171
                                i32.const 2
                                local.set 172
                                local.get 171
                                local.get 172
                                i32.shl
                                local.set 173
                                local.get 23
                                local.get 173
                                i32.add
                                local.set 174
                                local.get 174
                                f32.load
                                local.set 175
                                local.get 150
                                local.get 162
                                f32.mul
                                local.set 176
                                local.get 175
                                local.get 176
                                f32.add
                                local.set 177
                                i32.const 8112
                                local.set 178
                                local.get 88
                                local.get 178
                                i32.mul
                                local.set 179
                                i32.const 312
                                local.set 180
                                local.get 95
                                local.get 180
                                i32.mul
                                local.set 181
                                local.get 179
                                local.get 181
                                i32.add
                                local.set 182
                                i32.const 12
                                local.set 183
                                local.get 102
                                local.get 183
                                i32.mul
                                local.set 184
                                local.get 182
                                local.get 184
                                i32.add
                                local.set 185
                                local.get 185
                                local.get 109
                                i32.add
                                local.set 186
                                i32.const 2
                                local.set 187
                                local.get 186
                                local.get 187
                                i32.shl
                                local.set 188
                                local.get 23
                                local.get 188
                                i32.add
                                local.set 189
                                local.get 189
                                local.get 177
                                f32.store
                                i32.const 1
                                local.set 190
                                local.get 130
                                local.get 190
                                i32.add
                                local.set 191
                                local.get 191
                                local.set 129
                                br 0 (;@14;)
                              end
                            end
                            i32.const 1
                            local.set 192
                            local.get 123
                            local.get 192
                            i32.add
                            local.set 193
                            local.get 193
                            local.set 122
                            br 0 (;@12;)
                          end
                        end
                        i32.const 1
                        local.set 194
                        local.get 116
                        local.get 194
                        i32.add
                        local.set 195
                        local.get 195
                        local.set 115
                        br 0 (;@10;)
                      end
                    end
                    i32.const 1
                    local.set 196
                    local.get 109
                    local.get 196
                    i32.add
                    local.set 197
                    local.get 197
                    local.set 108
                    br 0 (;@8;)
                  end
                end
                i32.const 1
                local.set 198
                local.get 102
                local.get 198
                i32.add
                local.set 199
                local.get 199
                local.set 101
                br 0 (;@6;)
              end
            end
            i32.const 1
            local.set 200
            local.get 95
            local.get 200
            i32.add
            local.set 201
            local.get 201
            local.set 94
            br 0 (;@4;)
          end
        end
        i32.const 1
        local.set 202
        local.get 88
        local.get 202
        i32.add
        local.set 203
        local.get 203
        local.set 87
        br 0 (;@2;)
      end
    end
    local.get 0
    local.get 19
    i32.store
    local.get 0
    local.get 23
    i32.store offset=4
    local.get 0
    local.get 32
    i32.store offset=8
    local.get 0
    local.get 31
    i32.store offset=12
    local.get 0
    local.get 29
    i32.store offset=16
    local.get 0
    local.get 30
    i32.store offset=20
    local.get 0
    local.get 25
    i32.store offset=24
    local.get 0
    local.get 28
    i32.store offset=28
    local.get 0
    local.get 27
    i32.store offset=32
    local.get 0
    local.get 26
    i32.store offset=36
    local.get 0
    local.get 24
    i32.store offset=40
    return)
  (data $.L__constant_12xf32 (i32.const 0) "x\d4c\be\c3q\e1\bd-\e0\ac\bc\fex\f7\bcy\bc\9b\bd\ae|/>I\ccs\bdiw\90\bd\04yM=\b2\b3\93\bc\97+\ca=\04\9f\b3\bd")
  (data $.L__constant_12x3x3x1xf32 (i32.const 64) "o\d0\8d>\a16\b7>\cc%\98\beh\c8\c7\bdG\c2\89>C\15@>\88x\e8\bevCB>\12gZ>\95^\fd=mF\18\bd\a4%\f7\bd2\cfN=\13\109>\8e\fd->>\19+>\9d\c5S>\09an<\aa8\b1>:H\1c?\f8\8e\ff>\fd\a7\86\beYp*>\ca\c8\fa>T\dfX\bf\07\87L\bf\da\c3\ed\be\82\1bZ>9d\c8>#`\b9\be\b9\e0\17?\9c\1bX>\c3\85S\bf\e8~\16?\d7YH\be\cb\e9?\bf\d9\e65\bc\99Z\03=\d2[0\bb\e64'>)).>cyr>\f2=\c6=\bb\ce\89=\13\a6\0f\be\81`^\bf\dd\b8K\bf\fb2.\bfl\d7\88>\e3\8d\a1=sD\c9\be\cb\7f:?\8b\ac\d3>\07\c0\93>\bc\dc\98\be<V\86>\c1\fd3>z\0b\a7>\aen\9e=s\d0-\be\0b\06~>\92\1ed\bd\f4\d2\b4\bejU\e9\bd\f2\e04\bee\ab\a2\bc\ab??>Q\98\9b=m_\0e>e\91c>Z\9d\04>\9f\f7%>~\fa\0d?\bdk\d5\be\7f\d2\13\bf0\df\87\bfJ1\15\bf\01Y\ed=_[\07>\e5\e4$?}\86\0b?\cbP\9b>\bd\06T>\92\f2\d9>\85\15\c4\bd\a5TP<\bc\fb\bb>\9d]\ae\be-0\f3\be\c6\a5-\bfG\c7E\bf\c1oH>>\1c\16?kq>\bf\9bD\cb=lx\e4>\c4\b7:\bf\90M\96\bd\18/\04?jW8\be-\e4V>\89\18\a4\bc\d3\bb=>\9fd\91>\ed\95r=\5c\f9~\be\cb\cd8>\b8\c3\08>"))
