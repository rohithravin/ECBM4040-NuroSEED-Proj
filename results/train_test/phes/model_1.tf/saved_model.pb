Ż
Ġ
B
AddV2
x"T
y"T
z"T"
Ttype:
2	
h
Any	
input

reduction_indices"Tidx

output
"
	keep_dimsbool( "
Tidxtype0:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
+
IsInf
x"T
y
"
Ttype:
2
,
Log
x"T
y"T"
Ttype:

2


LogicalNot
x

y

q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
:
Maximum
x"T
y"T
z"T"
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp

OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisint˙˙˙˙˙˙˙˙˙"	
Ttype"
TItype0	:
2	
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
³
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
6
Pow
x"T
y"T
z"T"
Ttype:

2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
À
StatelessIf
cond"Tcond
input2Tin
output2Tout"
Tcondtype"
Tin
list(type)("
Tout
list(type)("
then_branchfunc"
else_branchfunc" 
output_shapeslist(shape)
 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ô8*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
ô8*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:*
dtype0

Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ô8*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m* 
_output_shapes
:
ô8*
dtype0

Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
x
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ô8*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v* 
_output_shapes
:
ô8*
dtype0

Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
x
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes	
:*
dtype0

NoOpNoOp
Ë 
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0* 
valueüBù Bò
˘
siamese_network
loss_tracker
	optimizer
loss
trainable_variables
regularization_losses
	variables
	keras_api
	
signatures
 

layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
trainable_variables
regularization_losses
	variables
	keras_api
4
	total
	count
	variables
	keras_api
d
iter

beta_1

beta_2
	decay
learning_ratemambvcvd
 

0
1
 

0
1
2
3
­
trainable_variables
metrics
layer_regularization_losses
regularization_losses

layers
	variables
 non_trainable_variables
!layer_metrics
 
 
 
 
"layer-0
#layer-1
$layer-2
%layer_with_weights-0
%layer-3
&trainable_variables
'regularization_losses
(	variables
)	keras_api
R
*trainable_variables
+regularization_losses
,	variables
-	keras_api

0
1
 

0
1
­
trainable_variables
.metrics
/layer_regularization_losses
regularization_losses

0layers
	variables
1non_trainable_variables
2layer_metrics
HF
VARIABLE_VALUEtotal-loss_tracker/total/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEcount-loss_tracker/count/.ATTRIBUTES/VARIABLE_VALUE

0
1

	variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_1/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense_1/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE

0
 

0

0
1


loss
R
3trainable_variables
4regularization_losses
5	variables
6	keras_api
R
7trainable_variables
8regularization_losses
9	variables
:	keras_api
R
;trainable_variables
<regularization_losses
=	variables
>	keras_api
h

kernel
bias
?trainable_variables
@regularization_losses
A	variables
B	keras_api

0
1
 

0
1
­
&trainable_variables
Cmetrics
Dlayer_regularization_losses
'regularization_losses

Elayers
(	variables
Fnon_trainable_variables
Glayer_metrics
 
 
 
­
*trainable_variables
Hmetrics
Ilayer_regularization_losses

Jlayers
+regularization_losses
,	variables
Knon_trainable_variables
Llayer_metrics
 
 


0
1
2
3
 
 
 
 
 
­
3trainable_variables
Mmetrics
Nlayer_regularization_losses

Olayers
4regularization_losses
5	variables
Pnon_trainable_variables
Qlayer_metrics
 
 
 
­
7trainable_variables
Rmetrics
Slayer_regularization_losses

Tlayers
8regularization_losses
9	variables
Unon_trainable_variables
Vlayer_metrics
 
 
 
­
;trainable_variables
Wmetrics
Xlayer_regularization_losses

Ylayers
<regularization_losses
=	variables
Znon_trainable_variables
[layer_metrics

0
1
 

0
1
­
?trainable_variables
\metrics
]layer_regularization_losses

^layers
@regularization_losses
A	variables
_non_trainable_variables
`layer_metrics
 
 

"0
#1
$2
%3
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
wu
VARIABLE_VALUEAdam/dense_1/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense_1/bias/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_1/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense_1/bias/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|
serving_default_input_1Placeholder*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
|
serving_default_input_2Placeholder*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
î
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2dense_1/kerneldense_1/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *.
f)R'
%__inference_signature_wrapper_5389818
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ù
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenametotal/Read/ReadVariableOpcount/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__traced_save_5390642
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenametotalcount	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratedense_1/kerneldense_1/biasAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/dense_1/kernel/vAdam/dense_1/bias/v*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference__traced_restore_5390691öÏ
¸
b
F__inference_flatten_1_layer_call_and_return_conditional_losses_5390555

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82

Identity"
identityIdentity:output:0*+
_input_shapes
:˙˙˙˙˙˙˙˙˙:T P
,
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

8
cond_true_5390352
cond_pow_add
cond_identity]

cond/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2

cond/pow/yl
cond/powPowcond_pow_addcond/pow/y:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

cond/pow]

cond/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2

cond/sub/yl
cond/subSubcond/pow:z:0cond/sub/y:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

cond/subZ
	cond/SqrtSqrtcond/sub:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
	cond/Sqrtg
cond/IdentityIdentitycond/Sqrt:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
cond/Identity"'
cond_identitycond/Identity:output:0*"
_input_shapes
:˙˙˙˙˙˙˙˙˙:) %
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

ġ
D__inference_model_1_layer_call_and_return_conditional_losses_5389657

inputs
inputs_1
sequential_1_5389647
sequential_1_5389649
identity˘$sequential_1/StatefulPartitionedCall˘&sequential_1/StatefulPartitionedCall_1Ż
$sequential_1/StatefulPartitionedCallStatefulPartitionedCallinputssequential_1_5389647sequential_1_5389649*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_53894962&
$sequential_1/StatefulPartitionedCallµ
&sequential_1/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1sequential_1_5389647sequential_1_5389649*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_53894962(
&sequential_1/StatefulPartitionedCall_1Ċ
 distance_layer_1/PartitionedCallPartitionedCall-sequential_1/StatefulPartitionedCall:output:0/sequential_1/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_distance_layer_1_layer_call_and_return_conditional_losses_53895912"
 distance_layer_1/PartitionedCallÉ
IdentityIdentity)distance_layer_1/PartitionedCall:output:0%^sequential_1/StatefulPartitionedCall'^sequential_1/StatefulPartitionedCall_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall2P
&sequential_1/StatefulPartitionedCall_1&sequential_1/StatefulPartitionedCall_1:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:PL
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ú@
 
D__inference_model_1_layer_call_and_return_conditional_losses_5390216
inputs_0
inputs_17
3sequential_1_dense_1_matmul_readvariableop_resource8
4sequential_1_dense_1_biasadd_readvariableop_resource
identity˘+sequential_1/dense_1/BiasAdd/ReadVariableOp˘-sequential_1/dense_1/BiasAdd_1/ReadVariableOp˘*sequential_1/dense_1/MatMul/ReadVariableOp˘,sequential_1/dense_1/MatMul_1/ReadVariableOp
$sequential_1/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2&
$sequential_1/dropout_1/dropout/Constğ
"sequential_1/dropout_1/dropout/MulMulinputs_0-sequential_1/dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2$
"sequential_1/dropout_1/dropout/Mul
$sequential_1/dropout_1/dropout/ShapeShapeinputs_0*
T0*
_output_shapes
:2&
$sequential_1/dropout_1/dropout/Shapeú
;sequential_1/dropout_1/dropout/random_uniform/RandomUniformRandomUniform-sequential_1/dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02=
;sequential_1/dropout_1/dropout/random_uniform/RandomUniform£
-sequential_1/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2/
-sequential_1/dropout_1/dropout/GreaterEqual/y
+sequential_1/dropout_1/dropout/GreaterEqualGreaterEqualDsequential_1/dropout_1/dropout/random_uniform/RandomUniform:output:06sequential_1/dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2-
+sequential_1/dropout_1/dropout/GreaterEqualĊ
#sequential_1/dropout_1/dropout/CastCast/sequential_1/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#sequential_1/dropout_1/dropout/Cast×
$sequential_1/dropout_1/dropout/Mul_1Mul&sequential_1/dropout_1/dropout/Mul:z:0'sequential_1/dropout_1/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2&
$sequential_1/dropout_1/dropout/Mul_1
5sequential_1/one_hot_encoding_layer_1/PartitionedCallPartitionedCall(sequential_1/dropout_1/dropout/Mul_1:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *!
fR
__inference_call_538927227
5sequential_1/one_hot_encoding_layer_1/PartitionedCall
sequential_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2
sequential_1/flatten_1/Constċ
sequential_1/flatten_1/ReshapeReshape>sequential_1/one_hot_encoding_layer_1/PartitionedCall:output:0%sequential_1/flatten_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82 
sequential_1/flatten_1/ReshapeÎ
*sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02,
*sequential_1/dense_1/MatMul/ReadVariableOpÔ
sequential_1/dense_1/MatMulMatMul'sequential_1/flatten_1/Reshape:output:02sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_1/dense_1/MatMulÌ
+sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+sequential_1/dense_1/BiasAdd/ReadVariableOpÖ
sequential_1/dense_1/BiasAddBiasAdd%sequential_1/dense_1/MatMul:product:03sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_1/dense_1/BiasAdd
&sequential_1/dropout_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2(
&sequential_1/dropout_1/dropout_1/ConstÁ
$sequential_1/dropout_1/dropout_1/MulMulinputs_1/sequential_1/dropout_1/dropout_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2&
$sequential_1/dropout_1/dropout_1/Mul
&sequential_1/dropout_1/dropout_1/ShapeShapeinputs_1*
T0*
_output_shapes
:2(
&sequential_1/dropout_1/dropout_1/Shape
=sequential_1/dropout_1/dropout_1/random_uniform/RandomUniformRandomUniform/sequential_1/dropout_1/dropout_1/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02?
=sequential_1/dropout_1/dropout_1/random_uniform/RandomUniform§
/sequential_1/dropout_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    21
/sequential_1/dropout_1/dropout_1/GreaterEqual/y£
-sequential_1/dropout_1/dropout_1/GreaterEqualGreaterEqualFsequential_1/dropout_1/dropout_1/random_uniform/RandomUniform:output:08sequential_1/dropout_1/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2/
-sequential_1/dropout_1/dropout_1/GreaterEqualË
%sequential_1/dropout_1/dropout_1/CastCast1sequential_1/dropout_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%sequential_1/dropout_1/dropout_1/Castß
&sequential_1/dropout_1/dropout_1/Mul_1Mul(sequential_1/dropout_1/dropout_1/Mul:z:0)sequential_1/dropout_1/dropout_1/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2(
&sequential_1/dropout_1/dropout_1/Mul_1
7sequential_1/one_hot_encoding_layer_1/PartitionedCall_1PartitionedCall*sequential_1/dropout_1/dropout_1/Mul_1:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *!
fR
__inference_call_538927229
7sequential_1/one_hot_encoding_layer_1/PartitionedCall_1
sequential_1/flatten_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2 
sequential_1/flatten_1/Const_1í
 sequential_1/flatten_1/Reshape_1Reshape@sequential_1/one_hot_encoding_layer_1/PartitionedCall_1:output:0'sequential_1/flatten_1/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82"
 sequential_1/flatten_1/Reshape_1Ò
,sequential_1/dense_1/MatMul_1/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02.
,sequential_1/dense_1/MatMul_1/ReadVariableOpÜ
sequential_1/dense_1/MatMul_1MatMul)sequential_1/flatten_1/Reshape_1:output:04sequential_1/dense_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_1/dense_1/MatMul_1?
-sequential_1/dense_1/BiasAdd_1/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-sequential_1/dense_1/BiasAdd_1/ReadVariableOpŜ
sequential_1/dense_1/BiasAdd_1BiasAdd'sequential_1/dense_1/MatMul_1:product:05sequential_1/dense_1/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
sequential_1/dense_1/BiasAdd_1
 distance_layer_1/PartitionedCallPartitionedCall%sequential_1/dense_1/BiasAdd:output:0'sequential_1/dense_1/BiasAdd_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *!
fR
__inference_call_53893462"
 distance_layer_1/PartitionedCall³
IdentityIdentity)distance_layer_1/PartitionedCall:output:0,^sequential_1/dense_1/BiasAdd/ReadVariableOp.^sequential_1/dense_1/BiasAdd_1/ReadVariableOp+^sequential_1/dense_1/MatMul/ReadVariableOp-^sequential_1/dense_1/MatMul_1/ReadVariableOp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2Z
+sequential_1/dense_1/BiasAdd/ReadVariableOp+sequential_1/dense_1/BiasAdd/ReadVariableOp2^
-sequential_1/dense_1/BiasAdd_1/ReadVariableOp-sequential_1/dense_1/BiasAdd_1/ReadVariableOp2X
*sequential_1/dense_1/MatMul/ReadVariableOp*sequential_1/dense_1/MatMul/ReadVariableOp2\
,sequential_1/dense_1/MatMul_1/ReadVariableOp,sequential_1/dense_1/MatMul_1/ReadVariableOp:R N
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/1
£9
Ñ
#__inference__traced_restore_5390691
file_prefix
assignvariableop_total
assignvariableop_1_count 
assignvariableop_2_adam_iter"
assignvariableop_3_adam_beta_1"
assignvariableop_4_adam_beta_2!
assignvariableop_5_adam_decay)
%assignvariableop_6_adam_learning_rate%
!assignvariableop_7_dense_1_kernel#
assignvariableop_8_dense_1_bias,
(assignvariableop_9_adam_dense_1_kernel_m+
'assignvariableop_10_adam_dense_1_bias_m-
)assignvariableop_11_adam_dense_1_kernel_v+
'assignvariableop_12_adam_dense_1_bias_v
identity_14˘AssignVariableOp˘AssignVariableOp_1˘AssignVariableOp_10˘AssignVariableOp_11˘AssignVariableOp_12˘AssignVariableOp_2˘AssignVariableOp_3˘AssignVariableOp_4˘AssignVariableOp_5˘AssignVariableOp_6˘AssignVariableOp_7˘AssignVariableOp_8˘AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB-loss_tracker/total/.ATTRIBUTES/VARIABLE_VALUEB-loss_tracker/count/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesŞ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesñ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*L
_output_shapes:
8::::::::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_totalIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1
AssignVariableOp_1AssignVariableOpassignvariableop_1_countIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2Ħ
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3£
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4£
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5˘
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Ş
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ĥ
AssignVariableOp_7AssignVariableOp!assignvariableop_7_dense_1_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¤
AssignVariableOp_8AssignVariableOpassignvariableop_8_dense_1_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9­
AssignVariableOp_9AssignVariableOp(assignvariableop_9_adam_dense_1_kernel_mIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ż
AssignVariableOp_10AssignVariableOp'assignvariableop_10_adam_dense_1_bias_mIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11ħ
AssignVariableOp_11AssignVariableOp)assignvariableop_11_adam_dense_1_kernel_vIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Ż
AssignVariableOp_12AssignVariableOp'assignvariableop_12_adam_dense_1_bias_vIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_129
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpü
Identity_13Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_13ï
Identity_14IdentityIdentity_13:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_14"#
identity_14Identity_14:output:0*I
_input_shapes8
6: :::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
î

.__inference_sequential_1_layer_call_fn_5390313

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallŭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_53894962
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
á

)__inference_model_1_layer_call_fn_5390260
inputs_0
inputs_1
unknown
	unknown_0
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_53896572
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/1

ġ
D__inference_model_1_layer_call_and_return_conditional_losses_5389633

inputs
inputs_1
sequential_1_5389623
sequential_1_5389625
identity˘$sequential_1/StatefulPartitionedCall˘&sequential_1/StatefulPartitionedCall_1Ż
$sequential_1/StatefulPartitionedCallStatefulPartitionedCallinputssequential_1_5389623sequential_1_5389625*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_53894752&
$sequential_1/StatefulPartitionedCallµ
&sequential_1/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1sequential_1_5389623sequential_1_5389625*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_53894752(
&sequential_1/StatefulPartitionedCall_1Ċ
 distance_layer_1/PartitionedCallPartitionedCall-sequential_1/StatefulPartitionedCall:output:0/sequential_1/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_distance_layer_1_layer_call_and_return_conditional_losses_53895912"
 distance_layer_1/PartitionedCallÉ
IdentityIdentity)distance_layer_1/PartitionedCall:output:0%^sequential_1/StatefulPartitionedCall'^sequential_1/StatefulPartitionedCall_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall2P
&sequential_1/StatefulPartitionedCall_1&sequential_1/StatefulPartitionedCall_1:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:PL
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
á

)__inference_model_1_layer_call_fn_5390250
inputs_0
inputs_1
unknown
	unknown_0
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_53896332
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/1
˙
/
__inference_call_5390549
x
identityY
CastCastx*

DstT0*

SrcT0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Casti
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
one_hot/on_valuek
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
one_hot/off_value`
one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :2
one_hot/depthĥ
one_hotOneHotCast:y:0one_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
one_hoti
IdentityIdentityone_hot:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:K G
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namex
ç

)__inference_model_1_layer_call_fn_5389664
	sequence1
	sequence2
unknown
	unknown_0
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCall	sequence1	sequence2unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_53896572
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:S O
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	sequence1:SO
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	sequence2

>
cond_false_5390052
cond_identity_add
cond_identityc
cond/IdentityIdentitycond_identity_add*
T0*
_output_shapes	
:2
cond/Identity"'
cond_identitycond/Identity:output:0*
_input_shapes	
::! 

_output_shapes	
:

G
+__inference_dropout_1_layer_call_fn_5390517

inputs
identityÈ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_53893772
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

e
F__inference_dropout_1_layer_call_and_return_conditional_losses_5389372

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/GreaterEqual/yż
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

8
cond_true_5389328
cond_pow_add
cond_identity]

cond/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2

cond/pow/yl
cond/powPowcond_pow_addcond/pow/y:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

cond/pow]

cond/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2

cond/sub/yl
cond/subSubcond/pow:z:0cond/sub/y:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

cond/subZ
	cond/SqrtSqrtcond/sub:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
	cond/Sqrtg
cond/IdentityIdentitycond/Sqrt:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
cond/Identity"'
cond_identitycond/Identity:output:0*"
_input_shapes
:˙˙˙˙˙˙˙˙˙:) %
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ĵ
l
U__inference_one_hot_encoding_layer_1_layer_call_and_return_conditional_losses_5389399
x
identityY
CastCastx*

DstT0*

SrcT0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Casti
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
one_hot/on_valuek
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
one_hot/off_value`
one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :2
one_hot/depthĥ
one_hotOneHotCast:y:0one_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
one_hoti
IdentityIdentityone_hot:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:K G
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namex
ö

I__inference_sequential_1_layer_call_and_return_conditional_losses_5390295

inputs*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity˘dense_1/BiasAdd/ReadVariableOp˘dense_1/MatMul/ReadVariableOpo
dropout_1/IdentityIdentityinputs*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout_1/Identityċ
(one_hot_encoding_layer_1/PartitionedCallPartitionedCalldropout_1/Identity:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *!
fR
__inference_call_53892722*
(one_hot_encoding_layer_1/PartitionedCalls
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2
flatten_1/Constħ
flatten_1/ReshapeReshape1one_hot_encoding_layer_1/PartitionedCall:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82
flatten_1/Reshape§
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02
dense_1/MatMul/ReadVariableOp 
dense_1/MatMulMatMulflatten_1/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp˘
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_1/BiasAdd?
IdentityIdentitydense_1/BiasAdd:output:0^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
?
>
cond_false_5389329
cond_identity_add
cond_identityk
cond/IdentityIdentitycond_identity_add*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
cond/Identity"'
cond_identitycond/Identity:output:0*"
_input_shapes
:˙˙˙˙˙˙˙˙˙:) %
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ì)

D__inference_model_1_layer_call_and_return_conditional_losses_5389730

inputs
inputs_17
3sequential_1_dense_1_matmul_readvariableop_resource8
4sequential_1_dense_1_biasadd_readvariableop_resource
identity˘+sequential_1/dense_1/BiasAdd/ReadVariableOp˘-sequential_1/dense_1/BiasAdd_1/ReadVariableOp˘*sequential_1/dense_1/MatMul/ReadVariableOp˘,sequential_1/dense_1/MatMul_1/ReadVariableOp
sequential_1/dropout_1/IdentityIdentityinputs*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2!
sequential_1/dropout_1/Identity
5sequential_1/one_hot_encoding_layer_1/PartitionedCallPartitionedCall(sequential_1/dropout_1/Identity:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *!
fR
__inference_call_538927227
5sequential_1/one_hot_encoding_layer_1/PartitionedCall
sequential_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2
sequential_1/flatten_1/Constċ
sequential_1/flatten_1/ReshapeReshape>sequential_1/one_hot_encoding_layer_1/PartitionedCall:output:0%sequential_1/flatten_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82 
sequential_1/flatten_1/ReshapeÎ
*sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02,
*sequential_1/dense_1/MatMul/ReadVariableOpÔ
sequential_1/dense_1/MatMulMatMul'sequential_1/flatten_1/Reshape:output:02sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_1/dense_1/MatMulÌ
+sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+sequential_1/dense_1/BiasAdd/ReadVariableOpÖ
sequential_1/dense_1/BiasAddBiasAdd%sequential_1/dense_1/MatMul:product:03sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_1/dense_1/BiasAdd
!sequential_1/dropout_1/Identity_1Identityinputs_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2#
!sequential_1/dropout_1/Identity_1
7sequential_1/one_hot_encoding_layer_1/PartitionedCall_1PartitionedCall*sequential_1/dropout_1/Identity_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *!
fR
__inference_call_538927229
7sequential_1/one_hot_encoding_layer_1/PartitionedCall_1
sequential_1/flatten_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2 
sequential_1/flatten_1/Const_1í
 sequential_1/flatten_1/Reshape_1Reshape@sequential_1/one_hot_encoding_layer_1/PartitionedCall_1:output:0'sequential_1/flatten_1/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82"
 sequential_1/flatten_1/Reshape_1Ò
,sequential_1/dense_1/MatMul_1/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02.
,sequential_1/dense_1/MatMul_1/ReadVariableOpÜ
sequential_1/dense_1/MatMul_1MatMul)sequential_1/flatten_1/Reshape_1:output:04sequential_1/dense_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_1/dense_1/MatMul_1?
-sequential_1/dense_1/BiasAdd_1/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-sequential_1/dense_1/BiasAdd_1/ReadVariableOpŜ
sequential_1/dense_1/BiasAdd_1BiasAdd'sequential_1/dense_1/MatMul_1:product:05sequential_1/dense_1/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
sequential_1/dense_1/BiasAdd_1
 distance_layer_1/PartitionedCallPartitionedCall%sequential_1/dense_1/BiasAdd:output:0'sequential_1/dense_1/BiasAdd_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *!
fR
__inference_call_53893462"
 distance_layer_1/PartitionedCall³
IdentityIdentity)distance_layer_1/PartitionedCall:output:0,^sequential_1/dense_1/BiasAdd/ReadVariableOp.^sequential_1/dense_1/BiasAdd_1/ReadVariableOp+^sequential_1/dense_1/MatMul/ReadVariableOp-^sequential_1/dense_1/MatMul_1/ReadVariableOp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2Z
+sequential_1/dense_1/BiasAdd/ReadVariableOp+sequential_1/dense_1/BiasAdd/ReadVariableOp2^
-sequential_1/dense_1/BiasAdd_1/ReadVariableOp-sequential_1/dense_1/BiasAdd_1/ReadVariableOp2X
*sequential_1/dense_1/MatMul/ReadVariableOp*sequential_1/dense_1/MatMul/ReadVariableOp2\
,sequential_1/dense_1/MatMul_1/ReadVariableOp,sequential_1/dense_1/MatMul_1/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:PL
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ĵ 
8
__inference_call_5390069
s1
s2
identityD
subSubs1s2*
T0* 
_output_shapes
:
2
subS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yU
powPowsub:z:0pow/y:output:0*
T0* 
_output_shapes
:
2
powy
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Sum/reduction_indices`
SumSumpow:z:0Sum/reduction_indices:output:0*
T0*
_output_shapes	
:2
SumW
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_1/yV
pow_1Pows2pow_1/y:output:0*
T0* 
_output_shapes
:
2
pow_1}
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Sum_1/reduction_indicesh
Sum_1Sum	pow_1:z:0 Sum_1/reduction_indices:output:0*
T0*
_output_shapes	
:2
Sum_1W
pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_2/yV
pow_2Pows1pow_2/y:output:0*
T0* 
_output_shapes
:
2
pow_2}
Sum_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Sum_2/reduction_indicesh
Sum_2Sum	pow_2:z:0 Sum_2/reduction_indices:output:0*
T0*
_output_shapes	
:2
Sum_2W
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
sub_1/x]
sub_1Subsub_1/x:output:0Sum_1:output:0*
T0*
_output_shapes	
:2
sub_1S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *½752
Const^
MaximumMaximum	sub_1:z:0Const:output:0*
T0*
_output_shapes	
:2	
MaximumW
sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
sub_2/x]
sub_2Subsub_2/x:output:0Sum_2:output:0*
T0*
_output_shapes	
:2
sub_2W
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *½752	
Const_1d
	Maximum_1Maximum	sub_2:z:0Const_1:output:0*
T0*
_output_shapes	
:2
	Maximum_1S
mulMulMaximum:z:0Maximum_1:z:0*
T0*
_output_shapes	
:2
mulW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *½752	
Const_2b
	Maximum_2Maximummul:z:0Const_2:output:0*
T0*
_output_shapes	
:2
	Maximum_2W
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
mul_1/x[
mul_1Mulmul_1/x:output:0Sum:output:0*
T0*
_output_shapes	
:2
mul_1]
truedivRealDiv	mul_1:z:0Maximum_2:z:0*
T0*
_output_shapes	
:2	
truedivS
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
add/xV
addAddV2add/x:output:0truediv:z:0*
T0*
_output_shapes	
:2
addW
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_3/yV
pow_3Powadd:z:0pow_3/y:output:0*
T0*
_output_shapes	
:2
pow_3H
IsInfIsInf	pow_3:z:0*
T0*
_output_shapes	
:2
IsInf\
Const_3Const*
_output_shapes
:*
dtype0*
valueB: 2	
Const_3F
AnyAny	IsInf:y:0Const_3:output:0*
_output_shapes
: 2
AnyL

LogicalNot
LogicalNotAny:output:0*
_output_shapes
: 2

LogicalNot§
condStatelessIfLogicalNot:y:0add:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*
_output_shapes	
:* 
_read_only_resource_inputs
 *%
else_branchR
cond_false_5390052*
output_shapes	
:*$
then_branchR
cond_true_53900512
cond_
cond/IdentityIdentitycond:output:0*
T0*
_output_shapes	
:2
cond/Identity^
add_1AddV2add:z:0cond/Identity:output:0*
T0*
_output_shapes	
:2
add_1B
LogLog	add_1:z:0*
T0*
_output_shapes	
:2
LogO
IdentityIdentityLog:y:0*
T0*
_output_shapes	
:2

Identity"
identityIdentity:output:0*+
_input_shapes
:
:
:D @
 
_output_shapes
:


_user_specified_names1:D@
 
_output_shapes
:


_user_specified_names2
×
/
__inference_call_5389995
x
identityQ
CastCastx*

DstT0*

SrcT0* 
_output_shapes
:
2
Casti
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
one_hot/on_valuek
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
one_hot/off_value`
one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :2
one_hot/depth?
one_hotOneHotCast:y:0one_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*$
_output_shapes
:2	
one_hota
IdentityIdentityone_hot:output:0*
T0*$
_output_shapes
:2

Identity"
identityIdentity:output:0*
_input_shapes
:
:C ?
 
_output_shapes
:


_user_specified_namex
?
>
cond_false_5390353
cond_identity_add
cond_identityk
cond/IdentityIdentitycond_identity_add*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
cond/Identity"'
cond_identitycond/Identity:output:0*"
_input_shapes
:˙˙˙˙˙˙˙˙˙:) %
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ŝ%
É
 __inference__traced_save_5390642
file_prefix$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop
savev2_const

identity_1˘MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardĤ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB-loss_tracker/total/.ATTRIBUTES/VARIABLE_VALUEB-loss_tracker/count/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names¤
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesï
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0 savev2_total_read_readvariableop savev2_count_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2ş
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesĦ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*^
_input_shapesM
K: : : : : : : : :
ô8::
ô8::
ô8:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
ô8:!	

_output_shapes	
::&
"
 
_output_shapes
:
ô8:!

_output_shapes	
::&"
 
_output_shapes
:
ô8:!

_output_shapes	
::

_output_shapes
: 
á

)__inference_model_1_layer_call_fn_5390178
inputs_0
inputs_1
unknown
	unknown_0
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_53897302
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/1

e
F__inference_dropout_1_layer_call_and_return_conditional_losses_5390502

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/GreaterEqual/yż
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ê,
˘
__inference_call_5390072
input_0
input_1?
;model_1_sequential_1_dense_1_matmul_readvariableop_resource@
<model_1_sequential_1_dense_1_biasadd_readvariableop_resource
identity˘3model_1/sequential_1/dense_1/BiasAdd/ReadVariableOp˘5model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOp˘2model_1/sequential_1/dense_1/MatMul/ReadVariableOp˘4model_1/sequential_1/dense_1/MatMul_1/ReadVariableOp
'model_1/sequential_1/dropout_1/IdentityIdentityinput_0*
T0* 
_output_shapes
:
2)
'model_1/sequential_1/dropout_1/Identity
=model_1/sequential_1/one_hot_encoding_layer_1/PartitionedCallPartitionedCall0model_1/sequential_1/dropout_1/Identity:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *!
fR
__inference_call_53899952?
=model_1/sequential_1/one_hot_encoding_layer_1/PartitionedCall
$model_1/sequential_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2&
$model_1/sequential_1/flatten_1/Constŭ
&model_1/sequential_1/flatten_1/ReshapeReshapeFmodel_1/sequential_1/one_hot_encoding_layer_1/PartitionedCall:output:0-model_1/sequential_1/flatten_1/Const:output:0*
T0* 
_output_shapes
:
ô82(
&model_1/sequential_1/flatten_1/Reshapeĉ
2model_1/sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp;model_1_sequential_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype024
2model_1/sequential_1/dense_1/MatMul/ReadVariableOpì
#model_1/sequential_1/dense_1/MatMulMatMul/model_1/sequential_1/flatten_1/Reshape:output:0:model_1/sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2%
#model_1/sequential_1/dense_1/MatMulä
3model_1/sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp<model_1_sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype025
3model_1/sequential_1/dense_1/BiasAdd/ReadVariableOpî
$model_1/sequential_1/dense_1/BiasAddBiasAdd-model_1/sequential_1/dense_1/MatMul:product:0;model_1/sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2&
$model_1/sequential_1/dense_1/BiasAdd
)model_1/sequential_1/dropout_1/Identity_1Identityinput_1*
T0* 
_output_shapes
:
2+
)model_1/sequential_1/dropout_1/Identity_1˘
?model_1/sequential_1/one_hot_encoding_layer_1/PartitionedCall_1PartitionedCall2model_1/sequential_1/dropout_1/Identity_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *!
fR
__inference_call_53899952A
?model_1/sequential_1/one_hot_encoding_layer_1/PartitionedCall_1Ħ
&model_1/sequential_1/flatten_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2(
&model_1/sequential_1/flatten_1/Const_1
(model_1/sequential_1/flatten_1/Reshape_1ReshapeHmodel_1/sequential_1/one_hot_encoding_layer_1/PartitionedCall_1:output:0/model_1/sequential_1/flatten_1/Const_1:output:0*
T0* 
_output_shapes
:
ô82*
(model_1/sequential_1/flatten_1/Reshape_1ê
4model_1/sequential_1/dense_1/MatMul_1/ReadVariableOpReadVariableOp;model_1_sequential_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype026
4model_1/sequential_1/dense_1/MatMul_1/ReadVariableOpô
%model_1/sequential_1/dense_1/MatMul_1MatMul1model_1/sequential_1/flatten_1/Reshape_1:output:0<model_1/sequential_1/dense_1/MatMul_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2'
%model_1/sequential_1/dense_1/MatMul_1è
5model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOpReadVariableOp<model_1_sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype027
5model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOpö
&model_1/sequential_1/dense_1/BiasAdd_1BiasAdd/model_1/sequential_1/dense_1/MatMul_1:product:0=model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2(
&model_1/sequential_1/dense_1/BiasAdd_1
(model_1/distance_layer_1/PartitionedCallPartitionedCall-model_1/sequential_1/dense_1/BiasAdd:output:0/model_1/sequential_1/dense_1/BiasAdd_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes	
:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *!
fR
__inference_call_53900692*
(model_1/distance_layer_1/PartitionedCallÓ
IdentityIdentity1model_1/distance_layer_1/PartitionedCall:output:04^model_1/sequential_1/dense_1/BiasAdd/ReadVariableOp6^model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOp3^model_1/sequential_1/dense_1/MatMul/ReadVariableOp5^model_1/sequential_1/dense_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes	
:2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :
:
::2j
3model_1/sequential_1/dense_1/BiasAdd/ReadVariableOp3model_1/sequential_1/dense_1/BiasAdd/ReadVariableOp2n
5model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOp5model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOp2h
2model_1/sequential_1/dense_1/MatMul/ReadVariableOp2model_1/sequential_1/dense_1/MatMul/ReadVariableOp2l
4model_1/sequential_1/dense_1/MatMul_1/ReadVariableOp4model_1/sequential_1/dense_1/MatMul_1/ReadVariableOp:I E
 
_output_shapes
:

!
_user_specified_name	input/0:IE
 
_output_shapes
:

!
_user_specified_name	input/1
?
>
cond_false_5390473
cond_identity_add
cond_identityk
cond/IdentityIdentitycond_identity_add*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
cond/Identity"'
cond_identitycond/Identity:output:0*"
_input_shapes
:˙˙˙˙˙˙˙˙˙:) %
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ı
Ĝ
I__inference_sequential_1_layer_call_and_return_conditional_losses_5389475

inputs
dense_1_5389469
dense_1_5389471
identity˘dense_1/StatefulPartitionedCall˘!dropout_1/StatefulPartitionedCallô
!dropout_1/StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_53893722#
!dropout_1/StatefulPartitionedCallħ
(one_hot_encoding_layer_1/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *^
fYRW
U__inference_one_hot_encoding_layer_1_layer_call_and_return_conditional_losses_53893992*
(one_hot_encoding_layer_1/PartitionedCall
flatten_1/PartitionedCallPartitionedCall1one_hot_encoding_layer_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô8* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_53894132
flatten_1/PartitionedCall²
dense_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_1_5389469dense_1_5389471*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_53894312!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
´
Q
:__inference_one_hot_encoding_layer_1_layer_call_fn_5390531
x
identityÖ
PartitionedCallPartitionedCallx*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *^
fYRW
U__inference_one_hot_encoding_layer_1_layer_call_and_return_conditional_losses_53893992
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:K G
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namex
.
Ö
L__inference_siamese_model_1_layer_call_and_return_conditional_losses_5389962
input_0
input_1?
;model_1_sequential_1_dense_1_matmul_readvariableop_resource@
<model_1_sequential_1_dense_1_biasadd_readvariableop_resource
identity˘3model_1/sequential_1/dense_1/BiasAdd/ReadVariableOp˘5model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOp˘2model_1/sequential_1/dense_1/MatMul/ReadVariableOp˘4model_1/sequential_1/dense_1/MatMul_1/ReadVariableOp
'model_1/sequential_1/dropout_1/IdentityIdentityinput_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2)
'model_1/sequential_1/dropout_1/Identity¤
=model_1/sequential_1/one_hot_encoding_layer_1/PartitionedCallPartitionedCall0model_1/sequential_1/dropout_1/Identity:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *!
fR
__inference_call_53892722?
=model_1/sequential_1/one_hot_encoding_layer_1/PartitionedCall
$model_1/sequential_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2&
$model_1/sequential_1/flatten_1/Const
&model_1/sequential_1/flatten_1/ReshapeReshapeFmodel_1/sequential_1/one_hot_encoding_layer_1/PartitionedCall:output:0-model_1/sequential_1/flatten_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82(
&model_1/sequential_1/flatten_1/Reshapeĉ
2model_1/sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp;model_1_sequential_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype024
2model_1/sequential_1/dense_1/MatMul/ReadVariableOpô
#model_1/sequential_1/dense_1/MatMulMatMul/model_1/sequential_1/flatten_1/Reshape:output:0:model_1/sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#model_1/sequential_1/dense_1/MatMulä
3model_1/sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp<model_1_sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype025
3model_1/sequential_1/dense_1/BiasAdd/ReadVariableOpö
$model_1/sequential_1/dense_1/BiasAddBiasAdd-model_1/sequential_1/dense_1/MatMul:product:0;model_1/sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2&
$model_1/sequential_1/dense_1/BiasAdd
)model_1/sequential_1/dropout_1/Identity_1Identityinput_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2+
)model_1/sequential_1/dropout_1/Identity_1Ş
?model_1/sequential_1/one_hot_encoding_layer_1/PartitionedCall_1PartitionedCall2model_1/sequential_1/dropout_1/Identity_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *!
fR
__inference_call_53892722A
?model_1/sequential_1/one_hot_encoding_layer_1/PartitionedCall_1Ħ
&model_1/sequential_1/flatten_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2(
&model_1/sequential_1/flatten_1/Const_1
(model_1/sequential_1/flatten_1/Reshape_1ReshapeHmodel_1/sequential_1/one_hot_encoding_layer_1/PartitionedCall_1:output:0/model_1/sequential_1/flatten_1/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82*
(model_1/sequential_1/flatten_1/Reshape_1ê
4model_1/sequential_1/dense_1/MatMul_1/ReadVariableOpReadVariableOp;model_1_sequential_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype026
4model_1/sequential_1/dense_1/MatMul_1/ReadVariableOpü
%model_1/sequential_1/dense_1/MatMul_1MatMul1model_1/sequential_1/flatten_1/Reshape_1:output:0<model_1/sequential_1/dense_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%model_1/sequential_1/dense_1/MatMul_1è
5model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOpReadVariableOp<model_1_sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype027
5model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOpŝ
&model_1/sequential_1/dense_1/BiasAdd_1BiasAdd/model_1/sequential_1/dense_1/MatMul_1:product:0=model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2(
&model_1/sequential_1/dense_1/BiasAdd_1 
(model_1/distance_layer_1/PartitionedCallPartitionedCall-model_1/sequential_1/dense_1/BiasAdd:output:0/model_1/sequential_1/dense_1/BiasAdd_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *!
fR
__inference_call_53893462*
(model_1/distance_layer_1/PartitionedCallÛ
IdentityIdentity1model_1/distance_layer_1/PartitionedCall:output:04^model_1/sequential_1/dense_1/BiasAdd/ReadVariableOp6^model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOp3^model_1/sequential_1/dense_1/MatMul/ReadVariableOp5^model_1/sequential_1/dense_1/MatMul_1/ReadVariableOp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2j
3model_1/sequential_1/dense_1/BiasAdd/ReadVariableOp3model_1/sequential_1/dense_1/BiasAdd/ReadVariableOp2n
5model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOp5model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOp2h
2model_1/sequential_1/dense_1/MatMul/ReadVariableOp2model_1/sequential_1/dense_1/MatMul/ReadVariableOp2l
4model_1/sequential_1/dense_1/MatMul_1/ReadVariableOp4model_1/sequential_1/dense_1/MatMul_1/ReadVariableOp:Q M
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input/0:QM
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input/1
Í
d
F__inference_dropout_1_layer_call_and_return_conditional_losses_5389377

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ĵ 
8
__inference_call_5390433
s1
s2
identityD
subSubs1s2*
T0* 
_output_shapes
:
2
subS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yU
powPowsub:z:0pow/y:output:0*
T0* 
_output_shapes
:
2
powy
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Sum/reduction_indices`
SumSumpow:z:0Sum/reduction_indices:output:0*
T0*
_output_shapes	
:2
SumW
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_1/yV
pow_1Pows2pow_1/y:output:0*
T0* 
_output_shapes
:
2
pow_1}
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Sum_1/reduction_indicesh
Sum_1Sum	pow_1:z:0 Sum_1/reduction_indices:output:0*
T0*
_output_shapes	
:2
Sum_1W
pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_2/yV
pow_2Pows1pow_2/y:output:0*
T0* 
_output_shapes
:
2
pow_2}
Sum_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Sum_2/reduction_indicesh
Sum_2Sum	pow_2:z:0 Sum_2/reduction_indices:output:0*
T0*
_output_shapes	
:2
Sum_2W
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
sub_1/x]
sub_1Subsub_1/x:output:0Sum_1:output:0*
T0*
_output_shapes	
:2
sub_1S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *½752
Const^
MaximumMaximum	sub_1:z:0Const:output:0*
T0*
_output_shapes	
:2	
MaximumW
sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
sub_2/x]
sub_2Subsub_2/x:output:0Sum_2:output:0*
T0*
_output_shapes	
:2
sub_2W
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *½752	
Const_1d
	Maximum_1Maximum	sub_2:z:0Const_1:output:0*
T0*
_output_shapes	
:2
	Maximum_1S
mulMulMaximum:z:0Maximum_1:z:0*
T0*
_output_shapes	
:2
mulW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *½752	
Const_2b
	Maximum_2Maximummul:z:0Const_2:output:0*
T0*
_output_shapes	
:2
	Maximum_2W
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
mul_1/x[
mul_1Mulmul_1/x:output:0Sum:output:0*
T0*
_output_shapes	
:2
mul_1]
truedivRealDiv	mul_1:z:0Maximum_2:z:0*
T0*
_output_shapes	
:2	
truedivS
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
add/xV
addAddV2add/x:output:0truediv:z:0*
T0*
_output_shapes	
:2
addW
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_3/yV
pow_3Powadd:z:0pow_3/y:output:0*
T0*
_output_shapes	
:2
pow_3H
IsInfIsInf	pow_3:z:0*
T0*
_output_shapes	
:2
IsInf\
Const_3Const*
_output_shapes
:*
dtype0*
valueB: 2	
Const_3F
AnyAny	IsInf:y:0Const_3:output:0*
_output_shapes
: 2
AnyL

LogicalNot
LogicalNotAny:output:0*
_output_shapes
: 2

LogicalNot§
condStatelessIfLogicalNot:y:0add:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*
_output_shapes	
:* 
_read_only_resource_inputs
 *%
else_branchR
cond_false_5390416*
output_shapes	
:*$
then_branchR
cond_true_53904152
cond_
cond/IdentityIdentitycond:output:0*
T0*
_output_shapes	
:2
cond/Identity^
add_1AddV2add:z:0cond/Identity:output:0*
T0*
_output_shapes	
:2
add_1B
LogLog	add_1:z:0*
T0*
_output_shapes	
:2
LogO
IdentityIdentityLog:y:0*
T0*
_output_shapes	
:2

Identity"
identityIdentity:output:0*+
_input_shapes
:
:
:D @
 
_output_shapes
:


_user_specified_names1:D@
 
_output_shapes
:


_user_specified_names2
?
>
cond_false_5389574
cond_identity_add
cond_identityk
cond/IdentityIdentitycond_identity_add*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
cond/Identity"'
cond_identitycond/Identity:output:0*"
_input_shapes
:˙˙˙˙˙˙˙˙˙:) %
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ô
³
"__inference__wrapped_model_5389356
input_1
input_2
siamese_model_1_5389350
siamese_model_1_5389352
identity˘'siamese_model_1/StatefulPartitionedCall
'siamese_model_1/StatefulPartitionedCallStatefulPartitionedCallinput_1input_2siamese_model_1_5389350siamese_model_1_5389352*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *!
fR
__inference_call_53893492)
'siamese_model_1/StatefulPartitionedCallŞ
IdentityIdentity0siamese_model_1/StatefulPartitionedCall:output:0(^siamese_model_1/StatefulPartitionedCall*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2R
'siamese_model_1/StatefulPartitionedCall'siamese_model_1/StatefulPartitionedCall:Q M
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1:QM
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_2
ë

1__inference_siamese_model_1_layer_call_fn_5389890
input_1
input_2
unknown
	unknown_0
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_siamese_model_1_layer_call_and_return_conditional_losses_53897812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1:QM
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_2
Í
d
F__inference_dropout_1_layer_call_and_return_conditional_losses_5390507

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ĵ
ù
D__inference_model_1_layer_call_and_return_conditional_losses_5389601
	sequence1
	sequence2
sequential_1_5389526
sequential_1_5389528
identity˘$sequential_1/StatefulPartitionedCall˘&sequential_1/StatefulPartitionedCall_1²
$sequential_1/StatefulPartitionedCallStatefulPartitionedCall	sequence1sequential_1_5389526sequential_1_5389528*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_53894752&
$sequential_1/StatefulPartitionedCallĥ
&sequential_1/StatefulPartitionedCall_1StatefulPartitionedCall	sequence2sequential_1_5389526sequential_1_5389528*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_53894752(
&sequential_1/StatefulPartitionedCall_1Ċ
 distance_layer_1/PartitionedCallPartitionedCall-sequential_1/StatefulPartitionedCall:output:0/sequential_1/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_distance_layer_1_layer_call_and_return_conditional_losses_53895912"
 distance_layer_1/PartitionedCallÉ
IdentityIdentity)distance_layer_1/PartitionedCall:output:0%^sequential_1/StatefulPartitionedCall'^sequential_1/StatefulPartitionedCall_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall2P
&sequential_1/StatefulPartitionedCall_1&sequential_1/StatefulPartitionedCall_1:S O
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	sequence1:SO
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	sequence2
µ

%__inference_signature_wrapper_5389818
input_1
input_2
unknown
	unknown_0
identity˘StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__wrapped_model_53893562
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1:QM
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_2
òG
Ö
L__inference_siamese_model_1_layer_call_and_return_conditional_losses_5389938
input_0
input_1?
;model_1_sequential_1_dense_1_matmul_readvariableop_resource@
<model_1_sequential_1_dense_1_biasadd_readvariableop_resource
identity˘3model_1/sequential_1/dense_1/BiasAdd/ReadVariableOp˘5model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOp˘2model_1/sequential_1/dense_1/MatMul/ReadVariableOp˘4model_1/sequential_1/dense_1/MatMul_1/ReadVariableOpĦ
,model_1/sequential_1/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2.
,model_1/sequential_1/dropout_1/dropout/ConstÒ
*model_1/sequential_1/dropout_1/dropout/MulMulinput_05model_1/sequential_1/dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2,
*model_1/sequential_1/dropout_1/dropout/Mul
,model_1/sequential_1/dropout_1/dropout/ShapeShapeinput_0*
T0*
_output_shapes
:2.
,model_1/sequential_1/dropout_1/dropout/Shape
Cmodel_1/sequential_1/dropout_1/dropout/random_uniform/RandomUniformRandomUniform5model_1/sequential_1/dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02E
Cmodel_1/sequential_1/dropout_1/dropout/random_uniform/RandomUniform³
5model_1/sequential_1/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5model_1/sequential_1/dropout_1/dropout/GreaterEqual/yğ
3model_1/sequential_1/dropout_1/dropout/GreaterEqualGreaterEqualLmodel_1/sequential_1/dropout_1/dropout/random_uniform/RandomUniform:output:0>model_1/sequential_1/dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙25
3model_1/sequential_1/dropout_1/dropout/GreaterEqualŬ
+model_1/sequential_1/dropout_1/dropout/CastCast7model_1/sequential_1/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2-
+model_1/sequential_1/dropout_1/dropout/Cast÷
,model_1/sequential_1/dropout_1/dropout/Mul_1Mul.model_1/sequential_1/dropout_1/dropout/Mul:z:0/model_1/sequential_1/dropout_1/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2.
,model_1/sequential_1/dropout_1/dropout/Mul_1¤
=model_1/sequential_1/one_hot_encoding_layer_1/PartitionedCallPartitionedCall0model_1/sequential_1/dropout_1/dropout/Mul_1:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *!
fR
__inference_call_53892722?
=model_1/sequential_1/one_hot_encoding_layer_1/PartitionedCall
$model_1/sequential_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2&
$model_1/sequential_1/flatten_1/Const
&model_1/sequential_1/flatten_1/ReshapeReshapeFmodel_1/sequential_1/one_hot_encoding_layer_1/PartitionedCall:output:0-model_1/sequential_1/flatten_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82(
&model_1/sequential_1/flatten_1/Reshapeĉ
2model_1/sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp;model_1_sequential_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype024
2model_1/sequential_1/dense_1/MatMul/ReadVariableOpô
#model_1/sequential_1/dense_1/MatMulMatMul/model_1/sequential_1/flatten_1/Reshape:output:0:model_1/sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#model_1/sequential_1/dense_1/MatMulä
3model_1/sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp<model_1_sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype025
3model_1/sequential_1/dense_1/BiasAdd/ReadVariableOpö
$model_1/sequential_1/dense_1/BiasAddBiasAdd-model_1/sequential_1/dense_1/MatMul:product:0;model_1/sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2&
$model_1/sequential_1/dense_1/BiasAdd?
.model_1/sequential_1/dropout_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?20
.model_1/sequential_1/dropout_1/dropout_1/ConstĜ
,model_1/sequential_1/dropout_1/dropout_1/MulMulinput_17model_1/sequential_1/dropout_1/dropout_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2.
,model_1/sequential_1/dropout_1/dropout_1/Mul
.model_1/sequential_1/dropout_1/dropout_1/ShapeShapeinput_1*
T0*
_output_shapes
:20
.model_1/sequential_1/dropout_1/dropout_1/Shape
Emodel_1/sequential_1/dropout_1/dropout_1/random_uniform/RandomUniformRandomUniform7model_1/sequential_1/dropout_1/dropout_1/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02G
Emodel_1/sequential_1/dropout_1/dropout_1/random_uniform/RandomUniform·
7model_1/sequential_1/dropout_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    29
7model_1/sequential_1/dropout_1/dropout_1/GreaterEqual/y?
5model_1/sequential_1/dropout_1/dropout_1/GreaterEqualGreaterEqualNmodel_1/sequential_1/dropout_1/dropout_1/random_uniform/RandomUniform:output:0@model_1/sequential_1/dropout_1/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙27
5model_1/sequential_1/dropout_1/dropout_1/GreaterEqual?
-model_1/sequential_1/dropout_1/dropout_1/CastCast9model_1/sequential_1/dropout_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2/
-model_1/sequential_1/dropout_1/dropout_1/Cast˙
.model_1/sequential_1/dropout_1/dropout_1/Mul_1Mul0model_1/sequential_1/dropout_1/dropout_1/Mul:z:01model_1/sequential_1/dropout_1/dropout_1/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙20
.model_1/sequential_1/dropout_1/dropout_1/Mul_1Ş
?model_1/sequential_1/one_hot_encoding_layer_1/PartitionedCall_1PartitionedCall2model_1/sequential_1/dropout_1/dropout_1/Mul_1:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *!
fR
__inference_call_53892722A
?model_1/sequential_1/one_hot_encoding_layer_1/PartitionedCall_1Ħ
&model_1/sequential_1/flatten_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2(
&model_1/sequential_1/flatten_1/Const_1
(model_1/sequential_1/flatten_1/Reshape_1ReshapeHmodel_1/sequential_1/one_hot_encoding_layer_1/PartitionedCall_1:output:0/model_1/sequential_1/flatten_1/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82*
(model_1/sequential_1/flatten_1/Reshape_1ê
4model_1/sequential_1/dense_1/MatMul_1/ReadVariableOpReadVariableOp;model_1_sequential_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype026
4model_1/sequential_1/dense_1/MatMul_1/ReadVariableOpü
%model_1/sequential_1/dense_1/MatMul_1MatMul1model_1/sequential_1/flatten_1/Reshape_1:output:0<model_1/sequential_1/dense_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%model_1/sequential_1/dense_1/MatMul_1è
5model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOpReadVariableOp<model_1_sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype027
5model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOpŝ
&model_1/sequential_1/dense_1/BiasAdd_1BiasAdd/model_1/sequential_1/dense_1/MatMul_1:product:0=model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2(
&model_1/sequential_1/dense_1/BiasAdd_1 
(model_1/distance_layer_1/PartitionedCallPartitionedCall-model_1/sequential_1/dense_1/BiasAdd:output:0/model_1/sequential_1/dense_1/BiasAdd_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *!
fR
__inference_call_53893462*
(model_1/distance_layer_1/PartitionedCallÛ
IdentityIdentity1model_1/distance_layer_1/PartitionedCall:output:04^model_1/sequential_1/dense_1/BiasAdd/ReadVariableOp6^model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOp3^model_1/sequential_1/dense_1/MatMul/ReadVariableOp5^model_1/sequential_1/dense_1/MatMul_1/ReadVariableOp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2j
3model_1/sequential_1/dense_1/BiasAdd/ReadVariableOp3model_1/sequential_1/dense_1/BiasAdd/ReadVariableOp2n
5model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOp5model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOp2h
2model_1/sequential_1/dense_1/MatMul/ReadVariableOp2model_1/sequential_1/dense_1/MatMul/ReadVariableOp2l
4model_1/sequential_1/dense_1/MatMul_1/ReadVariableOp4model_1/sequential_1/dense_1/MatMul_1/ReadVariableOp:Q M
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input/0:QM
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input/1
ñ

.__inference_sequential_1_layer_call_fn_5389482
input_2
unknown
	unknown_0
identity˘StatefulPartitionedCallŝ
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_53894752
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_2
¸
b
F__inference_flatten_1_layer_call_and_return_conditional_losses_5389413

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82

Identity"
identityIdentity:output:0*+
_input_shapes
:˙˙˙˙˙˙˙˙˙:T P
,
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

´
I__inference_sequential_1_layer_call_and_return_conditional_losses_5389496

inputs
dense_1_5389490
dense_1_5389492
identity˘dense_1/StatefulPartitionedCallÜ
dropout_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_53893772
dropout_1/PartitionedCallİ
(one_hot_encoding_layer_1/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *^
fYRW
U__inference_one_hot_encoding_layer_1_layer_call_and_return_conditional_losses_53893992*
(one_hot_encoding_layer_1/PartitionedCall
flatten_1/PartitionedCallPartitionedCall1one_hot_encoding_layer_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô8* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_53894132
flatten_1/PartitionedCall²
dense_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_1_5389490dense_1_5389492*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_53894312!
dense_1/StatefulPartitionedCall
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ú@
 
D__inference_model_1_layer_call_and_return_conditional_losses_5390134
inputs_0
inputs_17
3sequential_1_dense_1_matmul_readvariableop_resource8
4sequential_1_dense_1_biasadd_readvariableop_resource
identity˘+sequential_1/dense_1/BiasAdd/ReadVariableOp˘-sequential_1/dense_1/BiasAdd_1/ReadVariableOp˘*sequential_1/dense_1/MatMul/ReadVariableOp˘,sequential_1/dense_1/MatMul_1/ReadVariableOp
$sequential_1/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2&
$sequential_1/dropout_1/dropout/Constğ
"sequential_1/dropout_1/dropout/MulMulinputs_0-sequential_1/dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2$
"sequential_1/dropout_1/dropout/Mul
$sequential_1/dropout_1/dropout/ShapeShapeinputs_0*
T0*
_output_shapes
:2&
$sequential_1/dropout_1/dropout/Shapeú
;sequential_1/dropout_1/dropout/random_uniform/RandomUniformRandomUniform-sequential_1/dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02=
;sequential_1/dropout_1/dropout/random_uniform/RandomUniform£
-sequential_1/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2/
-sequential_1/dropout_1/dropout/GreaterEqual/y
+sequential_1/dropout_1/dropout/GreaterEqualGreaterEqualDsequential_1/dropout_1/dropout/random_uniform/RandomUniform:output:06sequential_1/dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2-
+sequential_1/dropout_1/dropout/GreaterEqualĊ
#sequential_1/dropout_1/dropout/CastCast/sequential_1/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#sequential_1/dropout_1/dropout/Cast×
$sequential_1/dropout_1/dropout/Mul_1Mul&sequential_1/dropout_1/dropout/Mul:z:0'sequential_1/dropout_1/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2&
$sequential_1/dropout_1/dropout/Mul_1
5sequential_1/one_hot_encoding_layer_1/PartitionedCallPartitionedCall(sequential_1/dropout_1/dropout/Mul_1:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *!
fR
__inference_call_538927227
5sequential_1/one_hot_encoding_layer_1/PartitionedCall
sequential_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2
sequential_1/flatten_1/Constċ
sequential_1/flatten_1/ReshapeReshape>sequential_1/one_hot_encoding_layer_1/PartitionedCall:output:0%sequential_1/flatten_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82 
sequential_1/flatten_1/ReshapeÎ
*sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02,
*sequential_1/dense_1/MatMul/ReadVariableOpÔ
sequential_1/dense_1/MatMulMatMul'sequential_1/flatten_1/Reshape:output:02sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_1/dense_1/MatMulÌ
+sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+sequential_1/dense_1/BiasAdd/ReadVariableOpÖ
sequential_1/dense_1/BiasAddBiasAdd%sequential_1/dense_1/MatMul:product:03sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_1/dense_1/BiasAdd
&sequential_1/dropout_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2(
&sequential_1/dropout_1/dropout_1/ConstÁ
$sequential_1/dropout_1/dropout_1/MulMulinputs_1/sequential_1/dropout_1/dropout_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2&
$sequential_1/dropout_1/dropout_1/Mul
&sequential_1/dropout_1/dropout_1/ShapeShapeinputs_1*
T0*
_output_shapes
:2(
&sequential_1/dropout_1/dropout_1/Shape
=sequential_1/dropout_1/dropout_1/random_uniform/RandomUniformRandomUniform/sequential_1/dropout_1/dropout_1/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02?
=sequential_1/dropout_1/dropout_1/random_uniform/RandomUniform§
/sequential_1/dropout_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    21
/sequential_1/dropout_1/dropout_1/GreaterEqual/y£
-sequential_1/dropout_1/dropout_1/GreaterEqualGreaterEqualFsequential_1/dropout_1/dropout_1/random_uniform/RandomUniform:output:08sequential_1/dropout_1/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2/
-sequential_1/dropout_1/dropout_1/GreaterEqualË
%sequential_1/dropout_1/dropout_1/CastCast1sequential_1/dropout_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%sequential_1/dropout_1/dropout_1/Castß
&sequential_1/dropout_1/dropout_1/Mul_1Mul(sequential_1/dropout_1/dropout_1/Mul:z:0)sequential_1/dropout_1/dropout_1/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2(
&sequential_1/dropout_1/dropout_1/Mul_1
7sequential_1/one_hot_encoding_layer_1/PartitionedCall_1PartitionedCall*sequential_1/dropout_1/dropout_1/Mul_1:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *!
fR
__inference_call_538927229
7sequential_1/one_hot_encoding_layer_1/PartitionedCall_1
sequential_1/flatten_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2 
sequential_1/flatten_1/Const_1í
 sequential_1/flatten_1/Reshape_1Reshape@sequential_1/one_hot_encoding_layer_1/PartitionedCall_1:output:0'sequential_1/flatten_1/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82"
 sequential_1/flatten_1/Reshape_1Ò
,sequential_1/dense_1/MatMul_1/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02.
,sequential_1/dense_1/MatMul_1/ReadVariableOpÜ
sequential_1/dense_1/MatMul_1MatMul)sequential_1/flatten_1/Reshape_1:output:04sequential_1/dense_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_1/dense_1/MatMul_1?
-sequential_1/dense_1/BiasAdd_1/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-sequential_1/dense_1/BiasAdd_1/ReadVariableOpŜ
sequential_1/dense_1/BiasAdd_1BiasAdd'sequential_1/dense_1/MatMul_1:product:05sequential_1/dense_1/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
sequential_1/dense_1/BiasAdd_1
 distance_layer_1/PartitionedCallPartitionedCall%sequential_1/dense_1/BiasAdd:output:0'sequential_1/dense_1/BiasAdd_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *!
fR
__inference_call_53893462"
 distance_layer_1/PartitionedCall³
IdentityIdentity)distance_layer_1/PartitionedCall:output:0,^sequential_1/dense_1/BiasAdd/ReadVariableOp.^sequential_1/dense_1/BiasAdd_1/ReadVariableOp+^sequential_1/dense_1/MatMul/ReadVariableOp-^sequential_1/dense_1/MatMul_1/ReadVariableOp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2Z
+sequential_1/dense_1/BiasAdd/ReadVariableOp+sequential_1/dense_1/BiasAdd/ReadVariableOp2^
-sequential_1/dense_1/BiasAdd_1/ReadVariableOp-sequential_1/dense_1/BiasAdd_1/ReadVariableOp2X
*sequential_1/dense_1/MatMul/ReadVariableOp*sequential_1/dense_1/MatMul/ReadVariableOp2\
,sequential_1/dense_1/MatMul_1/ReadVariableOp,sequential_1/dense_1/MatMul_1/ReadVariableOp:R N
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/1
×
/
__inference_call_5390540
x
identityQ
CastCastx*

DstT0*

SrcT0* 
_output_shapes
:
2
Casti
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
one_hot/on_valuek
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
one_hot/off_value`
one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :2
one_hot/depth?
one_hotOneHotCast:y:0one_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*$
_output_shapes
:2	
one_hota
IdentityIdentityone_hot:output:0*
T0*$
_output_shapes
:2

Identity"
identityIdentity:output:0*
_input_shapes
:
:C ?
 
_output_shapes
:


_user_specified_namex
Ñ"
m
M__inference_distance_layer_1_layer_call_and_return_conditional_losses_5390370
s1
s2
identityL
subSubs1s2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
subS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/y]
powPowsub:z:0pow/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
powy
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Sum/reduction_indicesh
SumSumpow:z:0Sum/reduction_indices:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
SumW
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_1/y^
pow_1Pows2pow_1/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pow_1}
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Sum_1/reduction_indicesp
Sum_1Sum	pow_1:z:0 Sum_1/reduction_indices:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Sum_1W
pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_2/y^
pow_2Pows1pow_2/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pow_2}
Sum_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Sum_2/reduction_indicesp
Sum_2Sum	pow_2:z:0 Sum_2/reduction_indices:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Sum_2W
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
sub_1/xe
sub_1Subsub_1/x:output:0Sum_1:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sub_1S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *½752
Constf
MaximumMaximum	sub_1:z:0Const:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
MaximumW
sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
sub_2/xe
sub_2Subsub_2/x:output:0Sum_2:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sub_2W
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *½752	
Const_1l
	Maximum_1Maximum	sub_2:z:0Const_1:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
	Maximum_1[
mulMulMaximum:z:0Maximum_1:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
mulW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *½752	
Const_2j
	Maximum_2Maximummul:z:0Const_2:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
	Maximum_2W
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
mul_1/xc
mul_1Mulmul_1/x:output:0Sum:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
mul_1e
truedivRealDiv	mul_1:z:0Maximum_2:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
truedivS
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
add/x^
addAddV2add/x:output:0truediv:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
addW
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_3/y^
pow_3Powadd:z:0pow_3/y:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pow_3P
IsInfIsInf	pow_3:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
IsInf\
Const_3Const*
_output_shapes
:*
dtype0*
valueB: 2	
Const_3F
AnyAny	IsInf:y:0Const_3:output:0*
_output_shapes
: 2
AnyL

LogicalNot
LogicalNotAny:output:0*
_output_shapes
: 2

LogicalNot·
condStatelessIfLogicalNot:y:0add:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *%
else_branchR
cond_false_5390353*"
output_shapes
:˙˙˙˙˙˙˙˙˙*$
then_branchR
cond_true_53903522
condg
cond/IdentityIdentitycond:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
cond/Identityf
add_1AddV2add:z:0cond/Identity:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
add_1J
LogLog	add_1:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
LogW
IdentityIdentityLog:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:L H
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_names1:LH
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_names2
ë

1__inference_siamese_model_1_layer_call_fn_5389900
input_1
input_2
unknown
	unknown_0
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_siamese_model_1_layer_call_and_return_conditional_losses_53897812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1:QM
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_2
?
~
)__inference_dense_1_layer_call_fn_5390579

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallĝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_53894312
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙ô8::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô8
 
_user_specified_nameinputs
.
Ö
L__inference_siamese_model_1_layer_call_and_return_conditional_losses_5389880
input_1
input_2?
;model_1_sequential_1_dense_1_matmul_readvariableop_resource@
<model_1_sequential_1_dense_1_biasadd_readvariableop_resource
identity˘3model_1/sequential_1/dense_1/BiasAdd/ReadVariableOp˘5model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOp˘2model_1/sequential_1/dense_1/MatMul/ReadVariableOp˘4model_1/sequential_1/dense_1/MatMul_1/ReadVariableOp
'model_1/sequential_1/dropout_1/IdentityIdentityinput_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2)
'model_1/sequential_1/dropout_1/Identity¤
=model_1/sequential_1/one_hot_encoding_layer_1/PartitionedCallPartitionedCall0model_1/sequential_1/dropout_1/Identity:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *!
fR
__inference_call_53892722?
=model_1/sequential_1/one_hot_encoding_layer_1/PartitionedCall
$model_1/sequential_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2&
$model_1/sequential_1/flatten_1/Const
&model_1/sequential_1/flatten_1/ReshapeReshapeFmodel_1/sequential_1/one_hot_encoding_layer_1/PartitionedCall:output:0-model_1/sequential_1/flatten_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82(
&model_1/sequential_1/flatten_1/Reshapeĉ
2model_1/sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp;model_1_sequential_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype024
2model_1/sequential_1/dense_1/MatMul/ReadVariableOpô
#model_1/sequential_1/dense_1/MatMulMatMul/model_1/sequential_1/flatten_1/Reshape:output:0:model_1/sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#model_1/sequential_1/dense_1/MatMulä
3model_1/sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp<model_1_sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype025
3model_1/sequential_1/dense_1/BiasAdd/ReadVariableOpö
$model_1/sequential_1/dense_1/BiasAddBiasAdd-model_1/sequential_1/dense_1/MatMul:product:0;model_1/sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2&
$model_1/sequential_1/dense_1/BiasAdd
)model_1/sequential_1/dropout_1/Identity_1Identityinput_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2+
)model_1/sequential_1/dropout_1/Identity_1Ş
?model_1/sequential_1/one_hot_encoding_layer_1/PartitionedCall_1PartitionedCall2model_1/sequential_1/dropout_1/Identity_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *!
fR
__inference_call_53892722A
?model_1/sequential_1/one_hot_encoding_layer_1/PartitionedCall_1Ħ
&model_1/sequential_1/flatten_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2(
&model_1/sequential_1/flatten_1/Const_1
(model_1/sequential_1/flatten_1/Reshape_1ReshapeHmodel_1/sequential_1/one_hot_encoding_layer_1/PartitionedCall_1:output:0/model_1/sequential_1/flatten_1/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82*
(model_1/sequential_1/flatten_1/Reshape_1ê
4model_1/sequential_1/dense_1/MatMul_1/ReadVariableOpReadVariableOp;model_1_sequential_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype026
4model_1/sequential_1/dense_1/MatMul_1/ReadVariableOpü
%model_1/sequential_1/dense_1/MatMul_1MatMul1model_1/sequential_1/flatten_1/Reshape_1:output:0<model_1/sequential_1/dense_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%model_1/sequential_1/dense_1/MatMul_1è
5model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOpReadVariableOp<model_1_sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype027
5model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOpŝ
&model_1/sequential_1/dense_1/BiasAdd_1BiasAdd/model_1/sequential_1/dense_1/MatMul_1:product:0=model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2(
&model_1/sequential_1/dense_1/BiasAdd_1 
(model_1/distance_layer_1/PartitionedCallPartitionedCall-model_1/sequential_1/dense_1/BiasAdd:output:0/model_1/sequential_1/dense_1/BiasAdd_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *!
fR
__inference_call_53893462*
(model_1/distance_layer_1/PartitionedCallÛ
IdentityIdentity1model_1/distance_layer_1/PartitionedCall:output:04^model_1/sequential_1/dense_1/BiasAdd/ReadVariableOp6^model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOp3^model_1/sequential_1/dense_1/MatMul/ReadVariableOp5^model_1/sequential_1/dense_1/MatMul_1/ReadVariableOp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2j
3model_1/sequential_1/dense_1/BiasAdd/ReadVariableOp3model_1/sequential_1/dense_1/BiasAdd/ReadVariableOp2n
5model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOp5model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOp2h
2model_1/sequential_1/dense_1/MatMul/ReadVariableOp2model_1/sequential_1/dense_1/MatMul/ReadVariableOp2l
4model_1/sequential_1/dense_1/MatMul_1/ReadVariableOp4model_1/sequential_1/dense_1/MatMul_1/ReadVariableOp:Q M
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1:QM
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_2
ë

1__inference_siamese_model_1_layer_call_fn_5389982
input_0
input_1
unknown
	unknown_0
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_0input_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_siamese_model_1_layer_call_and_return_conditional_losses_53897812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input/0:QM
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input/1
î

.__inference_sequential_1_layer_call_fn_5390304

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallŭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_53894752
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

8
cond_true_5390472
cond_pow_add
cond_identity]

cond/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2

cond/pow/yl
cond/powPowcond_pow_addcond/pow/y:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

cond/pow]

cond/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2

cond/sub/yl
cond/subSubcond/pow:z:0cond/sub/y:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

cond/subZ
	cond/SqrtSqrtcond/sub:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
	cond/Sqrtg
cond/IdentityIdentitycond/Sqrt:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
cond/Identity"'
cond_identitycond/Identity:output:0*"
_input_shapes
:˙˙˙˙˙˙˙˙˙:) %
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
	
Ŭ
D__inference_dense_1_layer_call_and_return_conditional_losses_5390570

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙ô8::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô8
 
_user_specified_nameinputs


I__inference_sequential_1_layer_call_and_return_conditional_losses_5390281

inputs*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity˘dense_1/BiasAdd/ReadVariableOp˘dense_1/MatMul/ReadVariableOpw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout_1/dropout/Const
dropout_1/dropout/MulMulinputs dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout_1/dropout/Mulh
dropout_1/dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout_1/dropout/ShapeÓ
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dropout_1/dropout/GreaterEqual/yç
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
dropout_1/dropout/GreaterEqual
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout_1/dropout/Cast£
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout_1/dropout/Mul_1ċ
(one_hot_encoding_layer_1/PartitionedCallPartitionedCalldropout_1/dropout/Mul_1:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *!
fR
__inference_call_53892722*
(one_hot_encoding_layer_1/PartitionedCalls
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2
flatten_1/Constħ
flatten_1/ReshapeReshape1one_hot_encoding_layer_1/PartitionedCall:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82
flatten_1/Reshape§
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02
dense_1/MatMul/ReadVariableOp 
dense_1/MatMulMatMulflatten_1/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp˘
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_1/BiasAdd?
IdentityIdentitydense_1/BiasAdd:output:0^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
	
Ŭ
D__inference_dense_1_layer_call_and_return_conditional_losses_5389431

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙ô8::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô8
 
_user_specified_nameinputs
?@

D__inference_model_1_layer_call_and_return_conditional_losses_5389706

inputs
inputs_17
3sequential_1_dense_1_matmul_readvariableop_resource8
4sequential_1_dense_1_biasadd_readvariableop_resource
identity˘+sequential_1/dense_1/BiasAdd/ReadVariableOp˘-sequential_1/dense_1/BiasAdd_1/ReadVariableOp˘*sequential_1/dense_1/MatMul/ReadVariableOp˘,sequential_1/dense_1/MatMul_1/ReadVariableOp
$sequential_1/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2&
$sequential_1/dropout_1/dropout/Constı
"sequential_1/dropout_1/dropout/MulMulinputs-sequential_1/dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2$
"sequential_1/dropout_1/dropout/Mul
$sequential_1/dropout_1/dropout/ShapeShapeinputs*
T0*
_output_shapes
:2&
$sequential_1/dropout_1/dropout/Shapeú
;sequential_1/dropout_1/dropout/random_uniform/RandomUniformRandomUniform-sequential_1/dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02=
;sequential_1/dropout_1/dropout/random_uniform/RandomUniform£
-sequential_1/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2/
-sequential_1/dropout_1/dropout/GreaterEqual/y
+sequential_1/dropout_1/dropout/GreaterEqualGreaterEqualDsequential_1/dropout_1/dropout/random_uniform/RandomUniform:output:06sequential_1/dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2-
+sequential_1/dropout_1/dropout/GreaterEqualĊ
#sequential_1/dropout_1/dropout/CastCast/sequential_1/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#sequential_1/dropout_1/dropout/Cast×
$sequential_1/dropout_1/dropout/Mul_1Mul&sequential_1/dropout_1/dropout/Mul:z:0'sequential_1/dropout_1/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2&
$sequential_1/dropout_1/dropout/Mul_1
5sequential_1/one_hot_encoding_layer_1/PartitionedCallPartitionedCall(sequential_1/dropout_1/dropout/Mul_1:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *!
fR
__inference_call_538927227
5sequential_1/one_hot_encoding_layer_1/PartitionedCall
sequential_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2
sequential_1/flatten_1/Constċ
sequential_1/flatten_1/ReshapeReshape>sequential_1/one_hot_encoding_layer_1/PartitionedCall:output:0%sequential_1/flatten_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82 
sequential_1/flatten_1/ReshapeÎ
*sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02,
*sequential_1/dense_1/MatMul/ReadVariableOpÔ
sequential_1/dense_1/MatMulMatMul'sequential_1/flatten_1/Reshape:output:02sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_1/dense_1/MatMulÌ
+sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+sequential_1/dense_1/BiasAdd/ReadVariableOpÖ
sequential_1/dense_1/BiasAddBiasAdd%sequential_1/dense_1/MatMul:product:03sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_1/dense_1/BiasAdd
&sequential_1/dropout_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2(
&sequential_1/dropout_1/dropout_1/ConstÁ
$sequential_1/dropout_1/dropout_1/MulMulinputs_1/sequential_1/dropout_1/dropout_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2&
$sequential_1/dropout_1/dropout_1/Mul
&sequential_1/dropout_1/dropout_1/ShapeShapeinputs_1*
T0*
_output_shapes
:2(
&sequential_1/dropout_1/dropout_1/Shape
=sequential_1/dropout_1/dropout_1/random_uniform/RandomUniformRandomUniform/sequential_1/dropout_1/dropout_1/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02?
=sequential_1/dropout_1/dropout_1/random_uniform/RandomUniform§
/sequential_1/dropout_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    21
/sequential_1/dropout_1/dropout_1/GreaterEqual/y£
-sequential_1/dropout_1/dropout_1/GreaterEqualGreaterEqualFsequential_1/dropout_1/dropout_1/random_uniform/RandomUniform:output:08sequential_1/dropout_1/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2/
-sequential_1/dropout_1/dropout_1/GreaterEqualË
%sequential_1/dropout_1/dropout_1/CastCast1sequential_1/dropout_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%sequential_1/dropout_1/dropout_1/Castß
&sequential_1/dropout_1/dropout_1/Mul_1Mul(sequential_1/dropout_1/dropout_1/Mul:z:0)sequential_1/dropout_1/dropout_1/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2(
&sequential_1/dropout_1/dropout_1/Mul_1
7sequential_1/one_hot_encoding_layer_1/PartitionedCall_1PartitionedCall*sequential_1/dropout_1/dropout_1/Mul_1:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *!
fR
__inference_call_538927229
7sequential_1/one_hot_encoding_layer_1/PartitionedCall_1
sequential_1/flatten_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2 
sequential_1/flatten_1/Const_1í
 sequential_1/flatten_1/Reshape_1Reshape@sequential_1/one_hot_encoding_layer_1/PartitionedCall_1:output:0'sequential_1/flatten_1/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82"
 sequential_1/flatten_1/Reshape_1Ò
,sequential_1/dense_1/MatMul_1/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02.
,sequential_1/dense_1/MatMul_1/ReadVariableOpÜ
sequential_1/dense_1/MatMul_1MatMul)sequential_1/flatten_1/Reshape_1:output:04sequential_1/dense_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_1/dense_1/MatMul_1?
-sequential_1/dense_1/BiasAdd_1/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-sequential_1/dense_1/BiasAdd_1/ReadVariableOpŜ
sequential_1/dense_1/BiasAdd_1BiasAdd'sequential_1/dense_1/MatMul_1:product:05sequential_1/dense_1/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
sequential_1/dense_1/BiasAdd_1
 distance_layer_1/PartitionedCallPartitionedCall%sequential_1/dense_1/BiasAdd:output:0'sequential_1/dense_1/BiasAdd_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *!
fR
__inference_call_53893462"
 distance_layer_1/PartitionedCall³
IdentityIdentity)distance_layer_1/PartitionedCall:output:0,^sequential_1/dense_1/BiasAdd/ReadVariableOp.^sequential_1/dense_1/BiasAdd_1/ReadVariableOp+^sequential_1/dense_1/MatMul/ReadVariableOp-^sequential_1/dense_1/MatMul_1/ReadVariableOp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2Z
+sequential_1/dense_1/BiasAdd/ReadVariableOp+sequential_1/dense_1/BiasAdd/ReadVariableOp2^
-sequential_1/dense_1/BiasAdd_1/ReadVariableOp-sequential_1/dense_1/BiasAdd_1/ReadVariableOp2X
*sequential_1/dense_1/MatMul/ReadVariableOp*sequential_1/dense_1/MatMul/ReadVariableOp2\
,sequential_1/dense_1/MatMul_1/ReadVariableOp,sequential_1/dense_1/MatMul_1/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:PL
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ĵ
Ù
I__inference_sequential_1_layer_call_and_return_conditional_losses_5389448
input_2
dense_1_5389442
dense_1_5389444
identity˘dense_1/StatefulPartitionedCall˘!dropout_1/StatefulPartitionedCallġ
!dropout_1/StatefulPartitionedCallStatefulPartitionedCallinput_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_53893722#
!dropout_1/StatefulPartitionedCallħ
(one_hot_encoding_layer_1/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *^
fYRW
U__inference_one_hot_encoding_layer_1_layer_call_and_return_conditional_losses_53893992*
(one_hot_encoding_layer_1/PartitionedCall
flatten_1/PartitionedCallPartitionedCall1one_hot_encoding_layer_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô8* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_53894132
flatten_1/PartitionedCall²
dense_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_1_5389442dense_1_5389444*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_53894312!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:Q M
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_2

R
2__inference_distance_layer_1_layer_call_fn_5390376
s1
s2
identityË
PartitionedCallPartitionedCalls1s2*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_distance_layer_1_layer_call_and_return_conditional_losses_53895912
PartitionedCallh
IdentityIdentityPartitionedCall:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:L H
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_names1:LH
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_names2
á
8
cond_true_5390415
cond_pow_add
cond_identity]

cond/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2

cond/pow/yd
cond/powPowcond_pow_addcond/pow/y:output:0*
T0*
_output_shapes	
:2

cond/pow]

cond/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2

cond/sub/yd
cond/subSubcond/pow:z:0cond/sub/y:output:0*
T0*
_output_shapes	
:2

cond/subR
	cond/SqrtSqrtcond/sub:z:0*
T0*
_output_shapes	
:2
	cond/Sqrt_
cond/IdentityIdentitycond/Sqrt:y:0*
T0*
_output_shapes	
:2
cond/Identity"'
cond_identitycond/Identity:output:0*
_input_shapes	
::! 

_output_shapes	
:
òG
Ö
L__inference_siamese_model_1_layer_call_and_return_conditional_losses_5389856
input_1
input_2?
;model_1_sequential_1_dense_1_matmul_readvariableop_resource@
<model_1_sequential_1_dense_1_biasadd_readvariableop_resource
identity˘3model_1/sequential_1/dense_1/BiasAdd/ReadVariableOp˘5model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOp˘2model_1/sequential_1/dense_1/MatMul/ReadVariableOp˘4model_1/sequential_1/dense_1/MatMul_1/ReadVariableOpĦ
,model_1/sequential_1/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2.
,model_1/sequential_1/dropout_1/dropout/ConstÒ
*model_1/sequential_1/dropout_1/dropout/MulMulinput_15model_1/sequential_1/dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2,
*model_1/sequential_1/dropout_1/dropout/Mul
,model_1/sequential_1/dropout_1/dropout/ShapeShapeinput_1*
T0*
_output_shapes
:2.
,model_1/sequential_1/dropout_1/dropout/Shape
Cmodel_1/sequential_1/dropout_1/dropout/random_uniform/RandomUniformRandomUniform5model_1/sequential_1/dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02E
Cmodel_1/sequential_1/dropout_1/dropout/random_uniform/RandomUniform³
5model_1/sequential_1/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5model_1/sequential_1/dropout_1/dropout/GreaterEqual/yğ
3model_1/sequential_1/dropout_1/dropout/GreaterEqualGreaterEqualLmodel_1/sequential_1/dropout_1/dropout/random_uniform/RandomUniform:output:0>model_1/sequential_1/dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙25
3model_1/sequential_1/dropout_1/dropout/GreaterEqualŬ
+model_1/sequential_1/dropout_1/dropout/CastCast7model_1/sequential_1/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2-
+model_1/sequential_1/dropout_1/dropout/Cast÷
,model_1/sequential_1/dropout_1/dropout/Mul_1Mul.model_1/sequential_1/dropout_1/dropout/Mul:z:0/model_1/sequential_1/dropout_1/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2.
,model_1/sequential_1/dropout_1/dropout/Mul_1¤
=model_1/sequential_1/one_hot_encoding_layer_1/PartitionedCallPartitionedCall0model_1/sequential_1/dropout_1/dropout/Mul_1:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *!
fR
__inference_call_53892722?
=model_1/sequential_1/one_hot_encoding_layer_1/PartitionedCall
$model_1/sequential_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2&
$model_1/sequential_1/flatten_1/Const
&model_1/sequential_1/flatten_1/ReshapeReshapeFmodel_1/sequential_1/one_hot_encoding_layer_1/PartitionedCall:output:0-model_1/sequential_1/flatten_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82(
&model_1/sequential_1/flatten_1/Reshapeĉ
2model_1/sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp;model_1_sequential_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype024
2model_1/sequential_1/dense_1/MatMul/ReadVariableOpô
#model_1/sequential_1/dense_1/MatMulMatMul/model_1/sequential_1/flatten_1/Reshape:output:0:model_1/sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#model_1/sequential_1/dense_1/MatMulä
3model_1/sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp<model_1_sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype025
3model_1/sequential_1/dense_1/BiasAdd/ReadVariableOpö
$model_1/sequential_1/dense_1/BiasAddBiasAdd-model_1/sequential_1/dense_1/MatMul:product:0;model_1/sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2&
$model_1/sequential_1/dense_1/BiasAdd?
.model_1/sequential_1/dropout_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?20
.model_1/sequential_1/dropout_1/dropout_1/ConstĜ
,model_1/sequential_1/dropout_1/dropout_1/MulMulinput_27model_1/sequential_1/dropout_1/dropout_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2.
,model_1/sequential_1/dropout_1/dropout_1/Mul
.model_1/sequential_1/dropout_1/dropout_1/ShapeShapeinput_2*
T0*
_output_shapes
:20
.model_1/sequential_1/dropout_1/dropout_1/Shape
Emodel_1/sequential_1/dropout_1/dropout_1/random_uniform/RandomUniformRandomUniform7model_1/sequential_1/dropout_1/dropout_1/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02G
Emodel_1/sequential_1/dropout_1/dropout_1/random_uniform/RandomUniform·
7model_1/sequential_1/dropout_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    29
7model_1/sequential_1/dropout_1/dropout_1/GreaterEqual/y?
5model_1/sequential_1/dropout_1/dropout_1/GreaterEqualGreaterEqualNmodel_1/sequential_1/dropout_1/dropout_1/random_uniform/RandomUniform:output:0@model_1/sequential_1/dropout_1/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙27
5model_1/sequential_1/dropout_1/dropout_1/GreaterEqual?
-model_1/sequential_1/dropout_1/dropout_1/CastCast9model_1/sequential_1/dropout_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2/
-model_1/sequential_1/dropout_1/dropout_1/Cast˙
.model_1/sequential_1/dropout_1/dropout_1/Mul_1Mul0model_1/sequential_1/dropout_1/dropout_1/Mul:z:01model_1/sequential_1/dropout_1/dropout_1/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙20
.model_1/sequential_1/dropout_1/dropout_1/Mul_1Ş
?model_1/sequential_1/one_hot_encoding_layer_1/PartitionedCall_1PartitionedCall2model_1/sequential_1/dropout_1/dropout_1/Mul_1:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *!
fR
__inference_call_53892722A
?model_1/sequential_1/one_hot_encoding_layer_1/PartitionedCall_1Ħ
&model_1/sequential_1/flatten_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2(
&model_1/sequential_1/flatten_1/Const_1
(model_1/sequential_1/flatten_1/Reshape_1ReshapeHmodel_1/sequential_1/one_hot_encoding_layer_1/PartitionedCall_1:output:0/model_1/sequential_1/flatten_1/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82*
(model_1/sequential_1/flatten_1/Reshape_1ê
4model_1/sequential_1/dense_1/MatMul_1/ReadVariableOpReadVariableOp;model_1_sequential_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype026
4model_1/sequential_1/dense_1/MatMul_1/ReadVariableOpü
%model_1/sequential_1/dense_1/MatMul_1MatMul1model_1/sequential_1/flatten_1/Reshape_1:output:0<model_1/sequential_1/dense_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%model_1/sequential_1/dense_1/MatMul_1è
5model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOpReadVariableOp<model_1_sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype027
5model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOpŝ
&model_1/sequential_1/dense_1/BiasAdd_1BiasAdd/model_1/sequential_1/dense_1/MatMul_1:product:0=model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2(
&model_1/sequential_1/dense_1/BiasAdd_1 
(model_1/distance_layer_1/PartitionedCallPartitionedCall-model_1/sequential_1/dense_1/BiasAdd:output:0/model_1/sequential_1/dense_1/BiasAdd_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *!
fR
__inference_call_53893462*
(model_1/distance_layer_1/PartitionedCallÛ
IdentityIdentity1model_1/distance_layer_1/PartitionedCall:output:04^model_1/sequential_1/dense_1/BiasAdd/ReadVariableOp6^model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOp3^model_1/sequential_1/dense_1/MatMul/ReadVariableOp5^model_1/sequential_1/dense_1/MatMul_1/ReadVariableOp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2j
3model_1/sequential_1/dense_1/BiasAdd/ReadVariableOp3model_1/sequential_1/dense_1/BiasAdd/ReadVariableOp2n
5model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOp5model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOp2h
2model_1/sequential_1/dense_1/MatMul/ReadVariableOp2model_1/sequential_1/dense_1/MatMul/ReadVariableOp2l
4model_1/sequential_1/dense_1/MatMul_1/ReadVariableOp4model_1/sequential_1/dense_1/MatMul_1/ReadVariableOp:Q M
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1:QM
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_2
˙
/
__inference_call_5389272
x
identityY
CastCastx*

DstT0*

SrcT0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Casti
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
one_hot/on_valuek
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
one_hot/off_value`
one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :2
one_hot/depthĥ
one_hotOneHotCast:y:0one_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
one_hoti
IdentityIdentityone_hot:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:K G
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namex
"
8
__inference_call_5390490
s1
s2
identityL
subSubs1s2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
subS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/y]
powPowsub:z:0pow/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
powy
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Sum/reduction_indicesh
SumSumpow:z:0Sum/reduction_indices:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
SumW
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_1/y^
pow_1Pows2pow_1/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pow_1}
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Sum_1/reduction_indicesp
Sum_1Sum	pow_1:z:0 Sum_1/reduction_indices:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Sum_1W
pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_2/y^
pow_2Pows1pow_2/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pow_2}
Sum_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Sum_2/reduction_indicesp
Sum_2Sum	pow_2:z:0 Sum_2/reduction_indices:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Sum_2W
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
sub_1/xe
sub_1Subsub_1/x:output:0Sum_1:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sub_1S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *½752
Constf
MaximumMaximum	sub_1:z:0Const:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
MaximumW
sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
sub_2/xe
sub_2Subsub_2/x:output:0Sum_2:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sub_2W
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *½752	
Const_1l
	Maximum_1Maximum	sub_2:z:0Const_1:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
	Maximum_1[
mulMulMaximum:z:0Maximum_1:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
mulW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *½752	
Const_2j
	Maximum_2Maximummul:z:0Const_2:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
	Maximum_2W
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
mul_1/xc
mul_1Mulmul_1/x:output:0Sum:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
mul_1e
truedivRealDiv	mul_1:z:0Maximum_2:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
truedivS
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
add/x^
addAddV2add/x:output:0truediv:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
addW
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_3/y^
pow_3Powadd:z:0pow_3/y:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pow_3P
IsInfIsInf	pow_3:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
IsInf\
Const_3Const*
_output_shapes
:*
dtype0*
valueB: 2	
Const_3F
AnyAny	IsInf:y:0Const_3:output:0*
_output_shapes
: 2
AnyL

LogicalNot
LogicalNotAny:output:0*
_output_shapes
: 2

LogicalNot·
condStatelessIfLogicalNot:y:0add:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *%
else_branchR
cond_false_5390473*"
output_shapes
:˙˙˙˙˙˙˙˙˙*$
then_branchR
cond_true_53904722
condg
cond/IdentityIdentitycond:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
cond/Identityf
add_1AddV2add:z:0cond/Identity:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
add_1J
LogLog	add_1:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
LogW
IdentityIdentityLog:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:L H
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_names1:LH
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_names2
ê-
˘
__inference_call_5390096
input_0
input_1?
;model_1_sequential_1_dense_1_matmul_readvariableop_resource@
<model_1_sequential_1_dense_1_biasadd_readvariableop_resource
identity˘3model_1/sequential_1/dense_1/BiasAdd/ReadVariableOp˘5model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOp˘2model_1/sequential_1/dense_1/MatMul/ReadVariableOp˘4model_1/sequential_1/dense_1/MatMul_1/ReadVariableOp
'model_1/sequential_1/dropout_1/IdentityIdentityinput_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2)
'model_1/sequential_1/dropout_1/Identity¤
=model_1/sequential_1/one_hot_encoding_layer_1/PartitionedCallPartitionedCall0model_1/sequential_1/dropout_1/Identity:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *!
fR
__inference_call_53892722?
=model_1/sequential_1/one_hot_encoding_layer_1/PartitionedCall
$model_1/sequential_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2&
$model_1/sequential_1/flatten_1/Const
&model_1/sequential_1/flatten_1/ReshapeReshapeFmodel_1/sequential_1/one_hot_encoding_layer_1/PartitionedCall:output:0-model_1/sequential_1/flatten_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82(
&model_1/sequential_1/flatten_1/Reshapeĉ
2model_1/sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp;model_1_sequential_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype024
2model_1/sequential_1/dense_1/MatMul/ReadVariableOpô
#model_1/sequential_1/dense_1/MatMulMatMul/model_1/sequential_1/flatten_1/Reshape:output:0:model_1/sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#model_1/sequential_1/dense_1/MatMulä
3model_1/sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp<model_1_sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype025
3model_1/sequential_1/dense_1/BiasAdd/ReadVariableOpö
$model_1/sequential_1/dense_1/BiasAddBiasAdd-model_1/sequential_1/dense_1/MatMul:product:0;model_1/sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2&
$model_1/sequential_1/dense_1/BiasAdd
)model_1/sequential_1/dropout_1/Identity_1Identityinput_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2+
)model_1/sequential_1/dropout_1/Identity_1Ş
?model_1/sequential_1/one_hot_encoding_layer_1/PartitionedCall_1PartitionedCall2model_1/sequential_1/dropout_1/Identity_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *!
fR
__inference_call_53892722A
?model_1/sequential_1/one_hot_encoding_layer_1/PartitionedCall_1Ħ
&model_1/sequential_1/flatten_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2(
&model_1/sequential_1/flatten_1/Const_1
(model_1/sequential_1/flatten_1/Reshape_1ReshapeHmodel_1/sequential_1/one_hot_encoding_layer_1/PartitionedCall_1:output:0/model_1/sequential_1/flatten_1/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82*
(model_1/sequential_1/flatten_1/Reshape_1ê
4model_1/sequential_1/dense_1/MatMul_1/ReadVariableOpReadVariableOp;model_1_sequential_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype026
4model_1/sequential_1/dense_1/MatMul_1/ReadVariableOpü
%model_1/sequential_1/dense_1/MatMul_1MatMul1model_1/sequential_1/flatten_1/Reshape_1:output:0<model_1/sequential_1/dense_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%model_1/sequential_1/dense_1/MatMul_1è
5model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOpReadVariableOp<model_1_sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype027
5model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOpŝ
&model_1/sequential_1/dense_1/BiasAdd_1BiasAdd/model_1/sequential_1/dense_1/MatMul_1:product:0=model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2(
&model_1/sequential_1/dense_1/BiasAdd_1 
(model_1/distance_layer_1/PartitionedCallPartitionedCall-model_1/sequential_1/dense_1/BiasAdd:output:0/model_1/sequential_1/dense_1/BiasAdd_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *!
fR
__inference_call_53893462*
(model_1/distance_layer_1/PartitionedCallÛ
IdentityIdentity1model_1/distance_layer_1/PartitionedCall:output:04^model_1/sequential_1/dense_1/BiasAdd/ReadVariableOp6^model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOp3^model_1/sequential_1/dense_1/MatMul/ReadVariableOp5^model_1/sequential_1/dense_1/MatMul_1/ReadVariableOp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2j
3model_1/sequential_1/dense_1/BiasAdd/ReadVariableOp3model_1/sequential_1/dense_1/BiasAdd/ReadVariableOp2n
5model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOp5model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOp2h
2model_1/sequential_1/dense_1/MatMul/ReadVariableOp2model_1/sequential_1/dense_1/MatMul/ReadVariableOp2l
4model_1/sequential_1/dense_1/MatMul_1/ReadVariableOp4model_1/sequential_1/dense_1/MatMul_1/ReadVariableOp:Q M
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input/0:QM
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input/1
ç

)__inference_model_1_layer_call_fn_5389640
	sequence1
	sequence2
unknown
	unknown_0
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCall	sequence1	sequence2unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_53896332
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:S O
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	sequence1:SO
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	sequence2
ë

1__inference_siamese_model_1_layer_call_fn_5389972
input_0
input_1
unknown
	unknown_0
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_0input_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_siamese_model_1_layer_call_and_return_conditional_losses_53897812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input/0:QM
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input/1
Ê
?
L__inference_siamese_model_1_layer_call_and_return_conditional_losses_5389781	
input
input_1
model_1_5389775
model_1_5389777
identity˘model_1/StatefulPartitionedCall
model_1/StatefulPartitionedCallStatefulPartitionedCallinputinput_1model_1_5389775model_1_5389777*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_53897302!
model_1/StatefulPartitionedCall
IdentityIdentity(model_1/StatefulPartitionedCall:output:0 ^model_1/StatefulPartitionedCall*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2B
model_1/StatefulPartitionedCallmodel_1/StatefulPartitionedCall:O K
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameinput:OK
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameinput
á

)__inference_model_1_layer_call_fn_5390168
inputs_0
inputs_1
unknown
	unknown_0
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_53897062
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/1
á
8
cond_true_5390051
cond_pow_add
cond_identity]

cond/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2

cond/pow/yd
cond/powPowcond_pow_addcond/pow/y:output:0*
T0*
_output_shapes	
:2

cond/pow]

cond/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2

cond/sub/yd
cond/subSubcond/pow:z:0cond/sub/y:output:0*
T0*
_output_shapes	
:2

cond/subR
	cond/SqrtSqrtcond/sub:z:0*
T0*
_output_shapes	
:2
	cond/Sqrt_
cond/IdentityIdentitycond/Sqrt:y:0*
T0*
_output_shapes	
:2
cond/Identity"'
cond_identitycond/Identity:output:0*
_input_shapes	
::! 

_output_shapes	
:
â-
 
__inference_call_5389349	
input
input_1?
;model_1_sequential_1_dense_1_matmul_readvariableop_resource@
<model_1_sequential_1_dense_1_biasadd_readvariableop_resource
identity˘3model_1/sequential_1/dense_1/BiasAdd/ReadVariableOp˘5model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOp˘2model_1/sequential_1/dense_1/MatMul/ReadVariableOp˘4model_1/sequential_1/dense_1/MatMul_1/ReadVariableOp
'model_1/sequential_1/dropout_1/IdentityIdentityinput*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2)
'model_1/sequential_1/dropout_1/Identity¤
=model_1/sequential_1/one_hot_encoding_layer_1/PartitionedCallPartitionedCall0model_1/sequential_1/dropout_1/Identity:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *!
fR
__inference_call_53892722?
=model_1/sequential_1/one_hot_encoding_layer_1/PartitionedCall
$model_1/sequential_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2&
$model_1/sequential_1/flatten_1/Const
&model_1/sequential_1/flatten_1/ReshapeReshapeFmodel_1/sequential_1/one_hot_encoding_layer_1/PartitionedCall:output:0-model_1/sequential_1/flatten_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82(
&model_1/sequential_1/flatten_1/Reshapeĉ
2model_1/sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp;model_1_sequential_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype024
2model_1/sequential_1/dense_1/MatMul/ReadVariableOpô
#model_1/sequential_1/dense_1/MatMulMatMul/model_1/sequential_1/flatten_1/Reshape:output:0:model_1/sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#model_1/sequential_1/dense_1/MatMulä
3model_1/sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp<model_1_sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype025
3model_1/sequential_1/dense_1/BiasAdd/ReadVariableOpö
$model_1/sequential_1/dense_1/BiasAddBiasAdd-model_1/sequential_1/dense_1/MatMul:product:0;model_1/sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2&
$model_1/sequential_1/dense_1/BiasAdd
)model_1/sequential_1/dropout_1/Identity_1Identityinput_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2+
)model_1/sequential_1/dropout_1/Identity_1Ş
?model_1/sequential_1/one_hot_encoding_layer_1/PartitionedCall_1PartitionedCall2model_1/sequential_1/dropout_1/Identity_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *!
fR
__inference_call_53892722A
?model_1/sequential_1/one_hot_encoding_layer_1/PartitionedCall_1Ħ
&model_1/sequential_1/flatten_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2(
&model_1/sequential_1/flatten_1/Const_1
(model_1/sequential_1/flatten_1/Reshape_1ReshapeHmodel_1/sequential_1/one_hot_encoding_layer_1/PartitionedCall_1:output:0/model_1/sequential_1/flatten_1/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82*
(model_1/sequential_1/flatten_1/Reshape_1ê
4model_1/sequential_1/dense_1/MatMul_1/ReadVariableOpReadVariableOp;model_1_sequential_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype026
4model_1/sequential_1/dense_1/MatMul_1/ReadVariableOpü
%model_1/sequential_1/dense_1/MatMul_1MatMul1model_1/sequential_1/flatten_1/Reshape_1:output:0<model_1/sequential_1/dense_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%model_1/sequential_1/dense_1/MatMul_1è
5model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOpReadVariableOp<model_1_sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype027
5model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOpŝ
&model_1/sequential_1/dense_1/BiasAdd_1BiasAdd/model_1/sequential_1/dense_1/MatMul_1:product:0=model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2(
&model_1/sequential_1/dense_1/BiasAdd_1 
(model_1/distance_layer_1/PartitionedCallPartitionedCall-model_1/sequential_1/dense_1/BiasAdd:output:0/model_1/sequential_1/dense_1/BiasAdd_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *!
fR
__inference_call_53893462*
(model_1/distance_layer_1/PartitionedCallÛ
IdentityIdentity1model_1/distance_layer_1/PartitionedCall:output:04^model_1/sequential_1/dense_1/BiasAdd/ReadVariableOp6^model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOp3^model_1/sequential_1/dense_1/MatMul/ReadVariableOp5^model_1/sequential_1/dense_1/MatMul_1/ReadVariableOp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2j
3model_1/sequential_1/dense_1/BiasAdd/ReadVariableOp3model_1/sequential_1/dense_1/BiasAdd/ReadVariableOp2n
5model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOp5model_1/sequential_1/dense_1/BiasAdd_1/ReadVariableOp2h
2model_1/sequential_1/dense_1/MatMul/ReadVariableOp2model_1/sequential_1/dense_1/MatMul/ReadVariableOp2l
4model_1/sequential_1/dense_1/MatMul_1/ReadVariableOp4model_1/sequential_1/dense_1/MatMul_1/ReadVariableOp:O K
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameinput:OK
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameinput
Ñ"
m
M__inference_distance_layer_1_layer_call_and_return_conditional_losses_5389591
s1
s2
identityL
subSubs1s2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
subS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/y]
powPowsub:z:0pow/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
powy
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Sum/reduction_indicesh
SumSumpow:z:0Sum/reduction_indices:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
SumW
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_1/y^
pow_1Pows2pow_1/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pow_1}
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Sum_1/reduction_indicesp
Sum_1Sum	pow_1:z:0 Sum_1/reduction_indices:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Sum_1W
pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_2/y^
pow_2Pows1pow_2/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pow_2}
Sum_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Sum_2/reduction_indicesp
Sum_2Sum	pow_2:z:0 Sum_2/reduction_indices:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Sum_2W
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
sub_1/xe
sub_1Subsub_1/x:output:0Sum_1:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sub_1S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *½752
Constf
MaximumMaximum	sub_1:z:0Const:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
MaximumW
sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
sub_2/xe
sub_2Subsub_2/x:output:0Sum_2:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sub_2W
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *½752	
Const_1l
	Maximum_1Maximum	sub_2:z:0Const_1:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
	Maximum_1[
mulMulMaximum:z:0Maximum_1:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
mulW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *½752	
Const_2j
	Maximum_2Maximummul:z:0Const_2:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
	Maximum_2W
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
mul_1/xc
mul_1Mulmul_1/x:output:0Sum:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
mul_1e
truedivRealDiv	mul_1:z:0Maximum_2:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
truedivS
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
add/x^
addAddV2add/x:output:0truediv:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
addW
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_3/y^
pow_3Powadd:z:0pow_3/y:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pow_3P
IsInfIsInf	pow_3:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
IsInf\
Const_3Const*
_output_shapes
:*
dtype0*
valueB: 2	
Const_3F
AnyAny	IsInf:y:0Const_3:output:0*
_output_shapes
: 2
AnyL

LogicalNot
LogicalNotAny:output:0*
_output_shapes
: 2

LogicalNot·
condStatelessIfLogicalNot:y:0add:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *%
else_branchR
cond_false_5389574*"
output_shapes
:˙˙˙˙˙˙˙˙˙*$
then_branchR
cond_true_53895732
condg
cond/IdentityIdentitycond:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
cond/Identityf
add_1AddV2add:z:0cond/Identity:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
add_1J
LogLog	add_1:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
LogW
IdentityIdentityLog:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:L H
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_names1:LH
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_names2

8
cond_true_5389573
cond_pow_add
cond_identity]

cond/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2

cond/pow/yl
cond/powPowcond_pow_addcond/pow/y:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

cond/pow]

cond/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2

cond/sub/yl
cond/subSubcond/pow:z:0cond/sub/y:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

cond/subZ
	cond/SqrtSqrtcond/sub:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
	cond/Sqrtg
cond/IdentityIdentitycond/Sqrt:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
cond/Identity"'
cond_identitycond/Identity:output:0*"
_input_shapes
:˙˙˙˙˙˙˙˙˙:) %
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ô)
 
D__inference_model_1_layer_call_and_return_conditional_losses_5390240
inputs_0
inputs_17
3sequential_1_dense_1_matmul_readvariableop_resource8
4sequential_1_dense_1_biasadd_readvariableop_resource
identity˘+sequential_1/dense_1/BiasAdd/ReadVariableOp˘-sequential_1/dense_1/BiasAdd_1/ReadVariableOp˘*sequential_1/dense_1/MatMul/ReadVariableOp˘,sequential_1/dense_1/MatMul_1/ReadVariableOp
sequential_1/dropout_1/IdentityIdentityinputs_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2!
sequential_1/dropout_1/Identity
5sequential_1/one_hot_encoding_layer_1/PartitionedCallPartitionedCall(sequential_1/dropout_1/Identity:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *!
fR
__inference_call_538927227
5sequential_1/one_hot_encoding_layer_1/PartitionedCall
sequential_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2
sequential_1/flatten_1/Constċ
sequential_1/flatten_1/ReshapeReshape>sequential_1/one_hot_encoding_layer_1/PartitionedCall:output:0%sequential_1/flatten_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82 
sequential_1/flatten_1/ReshapeÎ
*sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02,
*sequential_1/dense_1/MatMul/ReadVariableOpÔ
sequential_1/dense_1/MatMulMatMul'sequential_1/flatten_1/Reshape:output:02sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_1/dense_1/MatMulÌ
+sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+sequential_1/dense_1/BiasAdd/ReadVariableOpÖ
sequential_1/dense_1/BiasAddBiasAdd%sequential_1/dense_1/MatMul:product:03sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_1/dense_1/BiasAdd
!sequential_1/dropout_1/Identity_1Identityinputs_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2#
!sequential_1/dropout_1/Identity_1
7sequential_1/one_hot_encoding_layer_1/PartitionedCall_1PartitionedCall*sequential_1/dropout_1/Identity_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *!
fR
__inference_call_538927229
7sequential_1/one_hot_encoding_layer_1/PartitionedCall_1
sequential_1/flatten_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2 
sequential_1/flatten_1/Const_1í
 sequential_1/flatten_1/Reshape_1Reshape@sequential_1/one_hot_encoding_layer_1/PartitionedCall_1:output:0'sequential_1/flatten_1/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82"
 sequential_1/flatten_1/Reshape_1Ò
,sequential_1/dense_1/MatMul_1/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02.
,sequential_1/dense_1/MatMul_1/ReadVariableOpÜ
sequential_1/dense_1/MatMul_1MatMul)sequential_1/flatten_1/Reshape_1:output:04sequential_1/dense_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_1/dense_1/MatMul_1?
-sequential_1/dense_1/BiasAdd_1/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-sequential_1/dense_1/BiasAdd_1/ReadVariableOpŜ
sequential_1/dense_1/BiasAdd_1BiasAdd'sequential_1/dense_1/MatMul_1:product:05sequential_1/dense_1/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
sequential_1/dense_1/BiasAdd_1
 distance_layer_1/PartitionedCallPartitionedCall%sequential_1/dense_1/BiasAdd:output:0'sequential_1/dense_1/BiasAdd_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *!
fR
__inference_call_53893462"
 distance_layer_1/PartitionedCall³
IdentityIdentity)distance_layer_1/PartitionedCall:output:0,^sequential_1/dense_1/BiasAdd/ReadVariableOp.^sequential_1/dense_1/BiasAdd_1/ReadVariableOp+^sequential_1/dense_1/MatMul/ReadVariableOp-^sequential_1/dense_1/MatMul_1/ReadVariableOp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2Z
+sequential_1/dense_1/BiasAdd/ReadVariableOp+sequential_1/dense_1/BiasAdd/ReadVariableOp2^
-sequential_1/dense_1/BiasAdd_1/ReadVariableOp-sequential_1/dense_1/BiasAdd_1/ReadVariableOp2X
*sequential_1/dense_1/MatMul/ReadVariableOp*sequential_1/dense_1/MatMul/ReadVariableOp2\
,sequential_1/dense_1/MatMul_1/ReadVariableOp,sequential_1/dense_1/MatMul_1/ReadVariableOp:R N
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/1
?
G
+__inference_flatten_1_layer_call_fn_5390560

inputs
identityÈ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô8* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_53894132
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82

Identity"
identityIdentity:output:0*+
_input_shapes
:˙˙˙˙˙˙˙˙˙:T P
,
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
"
8
__inference_call_5389346
s1
s2
identityL
subSubs1s2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
subS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/y]
powPowsub:z:0pow/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
powy
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Sum/reduction_indicesh
SumSumpow:z:0Sum/reduction_indices:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
SumW
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_1/y^
pow_1Pows2pow_1/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pow_1}
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Sum_1/reduction_indicesp
Sum_1Sum	pow_1:z:0 Sum_1/reduction_indices:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Sum_1W
pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_2/y^
pow_2Pows1pow_2/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pow_2}
Sum_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Sum_2/reduction_indicesp
Sum_2Sum	pow_2:z:0 Sum_2/reduction_indices:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Sum_2W
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
sub_1/xe
sub_1Subsub_1/x:output:0Sum_1:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sub_1S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *½752
Constf
MaximumMaximum	sub_1:z:0Const:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
MaximumW
sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
sub_2/xe
sub_2Subsub_2/x:output:0Sum_2:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sub_2W
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *½752	
Const_1l
	Maximum_1Maximum	sub_2:z:0Const_1:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
	Maximum_1[
mulMulMaximum:z:0Maximum_1:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
mulW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *½752	
Const_2j
	Maximum_2Maximummul:z:0Const_2:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
	Maximum_2W
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
mul_1/xc
mul_1Mulmul_1/x:output:0Sum:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
mul_1e
truedivRealDiv	mul_1:z:0Maximum_2:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
truedivS
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
add/x^
addAddV2add/x:output:0truediv:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
addW
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_3/y^
pow_3Powadd:z:0pow_3/y:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pow_3P
IsInfIsInf	pow_3:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
IsInf\
Const_3Const*
_output_shapes
:*
dtype0*
valueB: 2	
Const_3F
AnyAny	IsInf:y:0Const_3:output:0*
_output_shapes
: 2
AnyL

LogicalNot
LogicalNotAny:output:0*
_output_shapes
: 2

LogicalNot·
condStatelessIfLogicalNot:y:0add:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *%
else_branchR
cond_false_5389329*"
output_shapes
:˙˙˙˙˙˙˙˙˙*$
then_branchR
cond_true_53893282
condg
cond/IdentityIdentitycond:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
cond/Identityf
add_1AddV2add:z:0cond/Identity:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
add_1J
LogLog	add_1:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
LogW
IdentityIdentityLog:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:L H
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_names1:LH
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_names2

>
cond_false_5390416
cond_identity_add
cond_identityc
cond/IdentityIdentitycond_identity_add*
T0*
_output_shapes	
:2
cond/Identity"'
cond_identitycond/Identity:output:0*
_input_shapes	
::! 

_output_shapes	
:
Ĵ
ù
D__inference_model_1_layer_call_and_return_conditional_losses_5389615
	sequence1
	sequence2
sequential_1_5389605
sequential_1_5389607
identity˘$sequential_1/StatefulPartitionedCall˘&sequential_1/StatefulPartitionedCall_1²
$sequential_1/StatefulPartitionedCallStatefulPartitionedCall	sequence1sequential_1_5389605sequential_1_5389607*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_53894962&
$sequential_1/StatefulPartitionedCallĥ
&sequential_1/StatefulPartitionedCall_1StatefulPartitionedCall	sequence2sequential_1_5389605sequential_1_5389607*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_53894962(
&sequential_1/StatefulPartitionedCall_1Ċ
 distance_layer_1/PartitionedCallPartitionedCall-sequential_1/StatefulPartitionedCall:output:0/sequential_1/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_distance_layer_1_layer_call_and_return_conditional_losses_53895912"
 distance_layer_1/PartitionedCallÉ
IdentityIdentity)distance_layer_1/PartitionedCall:output:0%^sequential_1/StatefulPartitionedCall'^sequential_1/StatefulPartitionedCall_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall2P
&sequential_1/StatefulPartitionedCall_1&sequential_1/StatefulPartitionedCall_1:S O
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	sequence1:SO
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	sequence2
İ
d
+__inference_dropout_1_layer_call_fn_5390512

inputs
identity˘StatefulPartitionedCallà
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_53893722
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*'
_input_shapes
:˙˙˙˙˙˙˙˙˙22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

µ
I__inference_sequential_1_layer_call_and_return_conditional_losses_5389460
input_2
dense_1_5389454
dense_1_5389456
identity˘dense_1/StatefulPartitionedCallŬ
dropout_1/PartitionedCallPartitionedCallinput_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_53893772
dropout_1/PartitionedCallİ
(one_hot_encoding_layer_1/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *^
fYRW
U__inference_one_hot_encoding_layer_1_layer_call_and_return_conditional_losses_53893992*
(one_hot_encoding_layer_1/PartitionedCall
flatten_1/PartitionedCallPartitionedCall1one_hot_encoding_layer_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô8* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_53894132
flatten_1/PartitionedCall²
dense_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_1_5389454dense_1_5389456*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_53894312!
dense_1/StatefulPartitionedCall
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:Q M
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_2
ĵ
l
U__inference_one_hot_encoding_layer_1_layer_call_and_return_conditional_losses_5390526
x
identityY
CastCastx*

DstT0*

SrcT0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Casti
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
one_hot/on_valuek
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
one_hot/off_value`
one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :2
one_hot/depthĥ
one_hotOneHotCast:y:0one_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
one_hoti
IdentityIdentityone_hot:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:K G
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namex
Ô)
 
D__inference_model_1_layer_call_and_return_conditional_losses_5390158
inputs_0
inputs_17
3sequential_1_dense_1_matmul_readvariableop_resource8
4sequential_1_dense_1_biasadd_readvariableop_resource
identity˘+sequential_1/dense_1/BiasAdd/ReadVariableOp˘-sequential_1/dense_1/BiasAdd_1/ReadVariableOp˘*sequential_1/dense_1/MatMul/ReadVariableOp˘,sequential_1/dense_1/MatMul_1/ReadVariableOp
sequential_1/dropout_1/IdentityIdentityinputs_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2!
sequential_1/dropout_1/Identity
5sequential_1/one_hot_encoding_layer_1/PartitionedCallPartitionedCall(sequential_1/dropout_1/Identity:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *!
fR
__inference_call_538927227
5sequential_1/one_hot_encoding_layer_1/PartitionedCall
sequential_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2
sequential_1/flatten_1/Constċ
sequential_1/flatten_1/ReshapeReshape>sequential_1/one_hot_encoding_layer_1/PartitionedCall:output:0%sequential_1/flatten_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82 
sequential_1/flatten_1/ReshapeÎ
*sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02,
*sequential_1/dense_1/MatMul/ReadVariableOpÔ
sequential_1/dense_1/MatMulMatMul'sequential_1/flatten_1/Reshape:output:02sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_1/dense_1/MatMulÌ
+sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+sequential_1/dense_1/BiasAdd/ReadVariableOpÖ
sequential_1/dense_1/BiasAddBiasAdd%sequential_1/dense_1/MatMul:product:03sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_1/dense_1/BiasAdd
!sequential_1/dropout_1/Identity_1Identityinputs_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2#
!sequential_1/dropout_1/Identity_1
7sequential_1/one_hot_encoding_layer_1/PartitionedCall_1PartitionedCall*sequential_1/dropout_1/Identity_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *!
fR
__inference_call_538927229
7sequential_1/one_hot_encoding_layer_1/PartitionedCall_1
sequential_1/flatten_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2 
sequential_1/flatten_1/Const_1í
 sequential_1/flatten_1/Reshape_1Reshape@sequential_1/one_hot_encoding_layer_1/PartitionedCall_1:output:0'sequential_1/flatten_1/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82"
 sequential_1/flatten_1/Reshape_1Ò
,sequential_1/dense_1/MatMul_1/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02.
,sequential_1/dense_1/MatMul_1/ReadVariableOpÜ
sequential_1/dense_1/MatMul_1MatMul)sequential_1/flatten_1/Reshape_1:output:04sequential_1/dense_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_1/dense_1/MatMul_1?
-sequential_1/dense_1/BiasAdd_1/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-sequential_1/dense_1/BiasAdd_1/ReadVariableOpŜ
sequential_1/dense_1/BiasAdd_1BiasAdd'sequential_1/dense_1/MatMul_1:product:05sequential_1/dense_1/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
sequential_1/dense_1/BiasAdd_1
 distance_layer_1/PartitionedCallPartitionedCall%sequential_1/dense_1/BiasAdd:output:0'sequential_1/dense_1/BiasAdd_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *!
fR
__inference_call_53893462"
 distance_layer_1/PartitionedCall³
IdentityIdentity)distance_layer_1/PartitionedCall:output:0,^sequential_1/dense_1/BiasAdd/ReadVariableOp.^sequential_1/dense_1/BiasAdd_1/ReadVariableOp+^sequential_1/dense_1/MatMul/ReadVariableOp-^sequential_1/dense_1/MatMul_1/ReadVariableOp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2Z
+sequential_1/dense_1/BiasAdd/ReadVariableOp+sequential_1/dense_1/BiasAdd/ReadVariableOp2^
-sequential_1/dense_1/BiasAdd_1/ReadVariableOp-sequential_1/dense_1/BiasAdd_1/ReadVariableOp2X
*sequential_1/dense_1/MatMul/ReadVariableOp*sequential_1/dense_1/MatMul/ReadVariableOp2\
,sequential_1/dense_1/MatMul_1/ReadVariableOp,sequential_1/dense_1/MatMul_1/ReadVariableOp:R N
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/1
ñ

.__inference_sequential_1_layer_call_fn_5389503
input_2
unknown
	unknown_0
identity˘StatefulPartitionedCallŝ
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_53894962
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_2"ħL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ĉ
serving_defaultÒ
<
input_11
serving_default_input_1:0˙˙˙˙˙˙˙˙˙
<
input_21
serving_default_input_2:0˙˙˙˙˙˙˙˙˙8
output_1,
StatefulPartitionedCall:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict:ë
Ì
siamese_network
loss_tracker
	optimizer
loss
trainable_variables
regularization_losses
	variables
	keras_api
	
signatures
e__call__
*f&call_and_return_all_conditional_losses
g_default_save_signature
hcall"Ĉ
_tf_keras_modelĴ{"class_name": "SiameseModel", "name": "siamese_model_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "SiameseModel"}, "training_config": {"loss": null, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "clipnorm": 1, "learning_rate": 1, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ê

layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
trainable_variables
regularization_losses
	variables
	keras_api
i__call__
*j&call_and_return_all_conditional_losses"
_tf_keras_networkñ{"class_name": "Functional", "name": "model_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1821]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "sequence1"}, "name": "sequence1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1821]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "sequence2"}, "name": "sequence2", "inbound_nodes": []}, {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1821]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}, {"class_name": "OneHotEncodingLayer", "config": {"layer was saved without config": true}}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 910, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "name": "sequential_1", "inbound_nodes": [[["sequence1", 0, 0, {}]], [["sequence2", 0, 0, {}]]]}, {"class_name": "DistanceLayer", "config": {"layer was saved without config": true}, "name": "distance_layer_1", "inbound_nodes": [[["sequential_1", 1, 0, {"s2": ["sequential_1", 2, 0]}]]]}], "input_layers": [["sequence1", 0, 0], ["sequence2", 0, 0]], "output_layers": {"class_name": "__tuple__", "items": [["distance_layer_1", 0, 0]]}}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1821]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1821]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1821]}, {"class_name": "TensorShape", "items": [null, 1821]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}}
ğ
	total
	count
	variables
	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
w
iter

beta_1

beta_2
	decay
learning_ratemambvcvd"
	optimizer
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
Ê
trainable_variables
metrics
layer_regularization_losses
regularization_losses

layers
	variables
 non_trainable_variables
!layer_metrics
e__call__
g_default_save_signature
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
,
kserving_default"
signature_map
ó"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "sequence1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1821]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1821]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "sequence1"}}
ó"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "sequence2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1821]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1821]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "sequence2"}}

"layer-0
#layer-1
$layer-2
%layer_with_weights-0
%layer-3
&trainable_variables
'regularization_losses
(	variables
)	keras_api
l__call__
*m&call_and_return_all_conditional_losses"¤
_tf_keras_sequential{"class_name": "Sequential", "name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1821]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}, {"class_name": "OneHotEncodingLayer", "config": {"layer was saved without config": true}}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 910, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1821]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential"}}
ı
*trainable_variables
+regularization_losses
,	variables
-	keras_api
n__call__
*o&call_and_return_all_conditional_losses
pcall" 
_tf_keras_layer{"class_name": "DistanceLayer", "name": "distance_layer_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
trainable_variables
.metrics
/layer_regularization_losses
regularization_losses

0layers
	variables
1non_trainable_variables
2layer_metrics
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
:  (2total
:  (2count
.
0
1"
trackable_list_wrapper
-
	variables"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
": 
ô82dense_1/kernel
:2dense_1/bias
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
*
loss"
trackable_dict_wrapper
ċ
3trainable_variables
4regularization_losses
5	variables
6	keras_api
q__call__
*r&call_and_return_all_conditional_losses"Ö
_tf_keras_layerĵ{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}
Ç
7trainable_variables
8regularization_losses
9	variables
:	keras_api
s__call__
*t&call_and_return_all_conditional_losses
ucall"?
_tf_keras_layer{"class_name": "OneHotEncodingLayer", "name": "one_hot_encoding_layer_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
ĉ
;trainable_variables
<regularization_losses
=	variables
>	keras_api
v__call__
*w&call_and_return_all_conditional_losses"×
_tf_keras_layer½{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
÷

kernel
bias
?trainable_variables
@regularization_losses
A	variables
B	keras_api
x__call__
*y&call_and_return_all_conditional_losses"Ò
_tf_keras_layer¸{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 910, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 7284}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7284]}}
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
&trainable_variables
Cmetrics
Dlayer_regularization_losses
'regularization_losses

Elayers
(	variables
Fnon_trainable_variables
Glayer_metrics
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
*trainable_variables
Hmetrics
Ilayer_regularization_losses

Jlayers
+regularization_losses
,	variables
Knon_trainable_variables
Llayer_metrics
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<

0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
3trainable_variables
Mmetrics
Nlayer_regularization_losses

Olayers
4regularization_losses
5	variables
Pnon_trainable_variables
Qlayer_metrics
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
7trainable_variables
Rmetrics
Slayer_regularization_losses

Tlayers
8regularization_losses
9	variables
Unon_trainable_variables
Vlayer_metrics
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
;trainable_variables
Wmetrics
Xlayer_regularization_losses

Ylayers
<regularization_losses
=	variables
Znon_trainable_variables
[layer_metrics
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
?trainable_variables
\metrics
]layer_regularization_losses

^layers
@regularization_losses
A	variables
_non_trainable_variables
`layer_metrics
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
"0
#1
$2
%3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
':%
ô82Adam/dense_1/kernel/m
 :2Adam/dense_1/bias/m
':%
ô82Adam/dense_1/kernel/v
 :2Adam/dense_1/bias/v
2
1__inference_siamese_model_1_layer_call_fn_5389890
1__inference_siamese_model_1_layer_call_fn_5389900
1__inference_siamese_model_1_layer_call_fn_5389972
1__inference_siamese_model_1_layer_call_fn_5389982²
İ²?
FullArgSpec(
args 
jself
jinput

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
?2í
L__inference_siamese_model_1_layer_call_and_return_conditional_losses_5389856
L__inference_siamese_model_1_layer_call_and_return_conditional_losses_5389962
L__inference_siamese_model_1_layer_call_and_return_conditional_losses_5389938
L__inference_siamese_model_1_layer_call_and_return_conditional_losses_5389880²
İ²?
FullArgSpec(
args 
jself
jinput

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
2
"__inference__wrapped_model_5389356à
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *P˘M
K˘H
"
input_1˙˙˙˙˙˙˙˙˙
"
input_2˙˙˙˙˙˙˙˙˙
Û2Ĝ
__inference_call_5390096
__inference_call_5390072Ħ
²
FullArgSpec
args
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
È2Ċ
)__inference_model_1_layer_call_fn_5390250
)__inference_model_1_layer_call_fn_5389664
)__inference_model_1_layer_call_fn_5390178
)__inference_model_1_layer_call_fn_5390168
)__inference_model_1_layer_call_fn_5389640
)__inference_model_1_layer_call_fn_5390260À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
ê2ç
D__inference_model_1_layer_call_and_return_conditional_losses_5389601
D__inference_model_1_layer_call_and_return_conditional_losses_5390158
D__inference_model_1_layer_call_and_return_conditional_losses_5390134
D__inference_model_1_layer_call_and_return_conditional_losses_5390216
D__inference_model_1_layer_call_and_return_conditional_losses_5389615
D__inference_model_1_layer_call_and_return_conditional_losses_5390240À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
ÓB?
%__inference_signature_wrapper_5389818input_1input_2"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
2
.__inference_sequential_1_layer_call_fn_5390304
.__inference_sequential_1_layer_call_fn_5389503
.__inference_sequential_1_layer_call_fn_5389482
.__inference_sequential_1_layer_call_fn_5390313À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
ò2ï
I__inference_sequential_1_layer_call_and_return_conditional_losses_5390281
I__inference_sequential_1_layer_call_and_return_conditional_losses_5390295
I__inference_sequential_1_layer_call_and_return_conditional_losses_5389448
I__inference_sequential_1_layer_call_and_return_conditional_losses_5389460À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Ŝ2Û
2__inference_distance_layer_1_layer_call_fn_5390376¤
²
FullArgSpec
args
jself
js1
js2
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ù2ö
M__inference_distance_layer_1_layer_call_and_return_conditional_losses_5390370¤
²
FullArgSpec
args
jself
js1
js2
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ŝ2Û
__inference_call_5390490
__inference_call_5390433¤
²
FullArgSpec
args
jself
js1
js2
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
2
+__inference_dropout_1_layer_call_fn_5390517
+__inference_dropout_1_layer_call_fn_5390512´
Ğ²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Ê2Ç
F__inference_dropout_1_layer_call_and_return_conditional_losses_5390502
F__inference_dropout_1_layer_call_and_return_conditional_losses_5390507´
Ğ²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
ß2Ü
:__inference_one_hot_encoding_layer_1_layer_call_fn_5390531
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ú2÷
U__inference_one_hot_encoding_layer_1_layer_call_and_return_conditional_losses_5390526
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
×2Ô
__inference_call_5390540
__inference_call_5390549
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ġ2Ò
+__inference_flatten_1_layer_call_fn_5390560˘
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
?2í
F__inference_flatten_1_layer_call_and_return_conditional_losses_5390555˘
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ó2?
)__inference_dense_1_layer_call_fn_5390579˘
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
î2ë
D__inference_dense_1_layer_call_and_return_conditional_losses_5390570˘
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 ¸
"__inference__wrapped_model_5389356Z˘W
P˘M
K˘H
"
input_1˙˙˙˙˙˙˙˙˙
"
input_2˙˙˙˙˙˙˙˙˙
Ş "/Ş,
*
output_1
output_1˙˙˙˙˙˙˙˙˙
__inference_call_5390072fJ˘G
@˘=
;˘8

input/0


input/1

Ş "˘

0
__inference_call_5390096~Z˘W
P˘M
K˘H
"
input/0˙˙˙˙˙˙˙˙˙
"
input/1˙˙˙˙˙˙˙˙˙
Ş "˘

0˙˙˙˙˙˙˙˙˙g
__inference_call_5390433K;˘8
1˘.

s1


s2

Ş "	
__inference_call_5390490cK˘H
A˘>

s1˙˙˙˙˙˙˙˙˙

s2˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙X
__inference_call_5390540<#˘ 
˘

x

Ş "h
__inference_call_5390549L+˘(
!˘

x˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙Ĥ
D__inference_dense_1_layer_call_and_return_conditional_losses_5390570^0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙ô8
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 ~
)__inference_dense_1_layer_call_fn_5390579Q0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙ô8
Ş "˙˙˙˙˙˙˙˙˙Á
M__inference_distance_layer_1_layer_call_and_return_conditional_losses_5390370pK˘H
A˘>

s1˙˙˙˙˙˙˙˙˙

s2˙˙˙˙˙˙˙˙˙
Ş "!˘

0˙˙˙˙˙˙˙˙˙
 
2__inference_distance_layer_1_layer_call_fn_5390376cK˘H
A˘>

s1˙˙˙˙˙˙˙˙˙

s2˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙¨
F__inference_dropout_1_layer_call_and_return_conditional_losses_5390502^4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 ¨
F__inference_dropout_1_layer_call_and_return_conditional_losses_5390507^4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 
+__inference_dropout_1_layer_call_fn_5390512Q4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p
Ş "˙˙˙˙˙˙˙˙˙
+__inference_dropout_1_layer_call_fn_5390517Q4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p 
Ş "˙˙˙˙˙˙˙˙˙¨
F__inference_flatten_1_layer_call_and_return_conditional_losses_5390555^4˘1
*˘'
%"
inputs˙˙˙˙˙˙˙˙˙
Ş "&˘#

0˙˙˙˙˙˙˙˙˙ô8
 
+__inference_flatten_1_layer_call_fn_5390560Q4˘1
*˘'
%"
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙ô8ß
D__inference_model_1_layer_call_and_return_conditional_losses_5389601f˘c
\˘Y
OL
$!
	sequence1˙˙˙˙˙˙˙˙˙
$!
	sequence2˙˙˙˙˙˙˙˙˙
p

 
Ş "(˘%
˘

0/0˙˙˙˙˙˙˙˙˙
 ß
D__inference_model_1_layer_call_and_return_conditional_losses_5389615f˘c
\˘Y
OL
$!
	sequence1˙˙˙˙˙˙˙˙˙
$!
	sequence2˙˙˙˙˙˙˙˙˙
p 

 
Ş "(˘%
˘

0/0˙˙˙˙˙˙˙˙˙
 Ŭ
D__inference_model_1_layer_call_and_return_conditional_losses_5390134d˘a
Z˘W
M˘J
# 
inputs/0˙˙˙˙˙˙˙˙˙
# 
inputs/1˙˙˙˙˙˙˙˙˙
p

 
Ş "(˘%
˘

0/0˙˙˙˙˙˙˙˙˙
 Ŭ
D__inference_model_1_layer_call_and_return_conditional_losses_5390158d˘a
Z˘W
M˘J
# 
inputs/0˙˙˙˙˙˙˙˙˙
# 
inputs/1˙˙˙˙˙˙˙˙˙
p 

 
Ş "(˘%
˘

0/0˙˙˙˙˙˙˙˙˙
 Ŭ
D__inference_model_1_layer_call_and_return_conditional_losses_5390216d˘a
Z˘W
MJ
# 
inputs/0˙˙˙˙˙˙˙˙˙
# 
inputs/1˙˙˙˙˙˙˙˙˙
p

 
Ş "(˘%
˘

0/0˙˙˙˙˙˙˙˙˙
 Ŭ
D__inference_model_1_layer_call_and_return_conditional_losses_5390240d˘a
Z˘W
MJ
# 
inputs/0˙˙˙˙˙˙˙˙˙
# 
inputs/1˙˙˙˙˙˙˙˙˙
p 

 
Ş "(˘%
˘

0/0˙˙˙˙˙˙˙˙˙
 ¸
)__inference_model_1_layer_call_fn_5389640f˘c
\˘Y
OL
$!
	sequence1˙˙˙˙˙˙˙˙˙
$!
	sequence2˙˙˙˙˙˙˙˙˙
p

 
Ş "˘

0˙˙˙˙˙˙˙˙˙¸
)__inference_model_1_layer_call_fn_5389664f˘c
\˘Y
OL
$!
	sequence1˙˙˙˙˙˙˙˙˙
$!
	sequence2˙˙˙˙˙˙˙˙˙
p 

 
Ş "˘

0˙˙˙˙˙˙˙˙˙ĥ
)__inference_model_1_layer_call_fn_5390168d˘a
Z˘W
M˘J
# 
inputs/0˙˙˙˙˙˙˙˙˙
# 
inputs/1˙˙˙˙˙˙˙˙˙
p

 
Ş "˘

0˙˙˙˙˙˙˙˙˙ĥ
)__inference_model_1_layer_call_fn_5390178d˘a
Z˘W
M˘J
# 
inputs/0˙˙˙˙˙˙˙˙˙
# 
inputs/1˙˙˙˙˙˙˙˙˙
p 

 
Ş "˘

0˙˙˙˙˙˙˙˙˙ĥ
)__inference_model_1_layer_call_fn_5390250d˘a
Z˘W
MJ
# 
inputs/0˙˙˙˙˙˙˙˙˙
# 
inputs/1˙˙˙˙˙˙˙˙˙
p

 
Ş "˘

0˙˙˙˙˙˙˙˙˙ĥ
)__inference_model_1_layer_call_fn_5390260d˘a
Z˘W
MJ
# 
inputs/0˙˙˙˙˙˙˙˙˙
# 
inputs/1˙˙˙˙˙˙˙˙˙
p 

 
Ş "˘

0˙˙˙˙˙˙˙˙˙²
U__inference_one_hot_encoding_layer_1_layer_call_and_return_conditional_losses_5390526Y+˘(
!˘

x˙˙˙˙˙˙˙˙˙
Ş "*˘'
 
0˙˙˙˙˙˙˙˙˙
 
:__inference_one_hot_encoding_layer_1_layer_call_fn_5390531L+˘(
!˘

x˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙´
I__inference_sequential_1_layer_call_and_return_conditional_losses_5389448g9˘6
/˘,
"
input_2˙˙˙˙˙˙˙˙˙
p

 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 ´
I__inference_sequential_1_layer_call_and_return_conditional_losses_5389460g9˘6
/˘,
"
input_2˙˙˙˙˙˙˙˙˙
p 

 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 ³
I__inference_sequential_1_layer_call_and_return_conditional_losses_5390281f8˘5
.˘+
!
inputs˙˙˙˙˙˙˙˙˙
p

 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 ³
I__inference_sequential_1_layer_call_and_return_conditional_losses_5390295f8˘5
.˘+
!
inputs˙˙˙˙˙˙˙˙˙
p 

 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 
.__inference_sequential_1_layer_call_fn_5389482Z9˘6
/˘,
"
input_2˙˙˙˙˙˙˙˙˙
p

 
Ş "˙˙˙˙˙˙˙˙˙
.__inference_sequential_1_layer_call_fn_5389503Z9˘6
/˘,
"
input_2˙˙˙˙˙˙˙˙˙
p 

 
Ş "˙˙˙˙˙˙˙˙˙
.__inference_sequential_1_layer_call_fn_5390304Y8˘5
.˘+
!
inputs˙˙˙˙˙˙˙˙˙
p

 
Ş "˙˙˙˙˙˙˙˙˙
.__inference_sequential_1_layer_call_fn_5390313Y8˘5
.˘+
!
inputs˙˙˙˙˙˙˙˙˙
p 

 
Ş "˙˙˙˙˙˙˙˙˙ß
L__inference_siamese_model_1_layer_call_and_return_conditional_losses_5389856^˘[
T˘Q
K˘H
"
input_1˙˙˙˙˙˙˙˙˙
"
input_2˙˙˙˙˙˙˙˙˙
p
Ş "(˘%
˘

0/0˙˙˙˙˙˙˙˙˙
 ß
L__inference_siamese_model_1_layer_call_and_return_conditional_losses_5389880^˘[
T˘Q
K˘H
"
input_1˙˙˙˙˙˙˙˙˙
"
input_2˙˙˙˙˙˙˙˙˙
p 
Ş "(˘%
˘

0/0˙˙˙˙˙˙˙˙˙
 ß
L__inference_siamese_model_1_layer_call_and_return_conditional_losses_5389938^˘[
T˘Q
K˘H
"
input/0˙˙˙˙˙˙˙˙˙
"
input/1˙˙˙˙˙˙˙˙˙
p
Ş "(˘%
˘

0/0˙˙˙˙˙˙˙˙˙
 ß
L__inference_siamese_model_1_layer_call_and_return_conditional_losses_5389962^˘[
T˘Q
K˘H
"
input/0˙˙˙˙˙˙˙˙˙
"
input/1˙˙˙˙˙˙˙˙˙
p 
Ş "(˘%
˘

0/0˙˙˙˙˙˙˙˙˙
 ¸
1__inference_siamese_model_1_layer_call_fn_5389890^˘[
T˘Q
K˘H
"
input_1˙˙˙˙˙˙˙˙˙
"
input_2˙˙˙˙˙˙˙˙˙
p
Ş "˘

0˙˙˙˙˙˙˙˙˙¸
1__inference_siamese_model_1_layer_call_fn_5389900^˘[
T˘Q
K˘H
"
input_1˙˙˙˙˙˙˙˙˙
"
input_2˙˙˙˙˙˙˙˙˙
p 
Ş "˘

0˙˙˙˙˙˙˙˙˙¸
1__inference_siamese_model_1_layer_call_fn_5389972^˘[
T˘Q
K˘H
"
input/0˙˙˙˙˙˙˙˙˙
"
input/1˙˙˙˙˙˙˙˙˙
p
Ş "˘

0˙˙˙˙˙˙˙˙˙¸
1__inference_siamese_model_1_layer_call_fn_5389982^˘[
T˘Q
K˘H
"
input/0˙˙˙˙˙˙˙˙˙
"
input/1˙˙˙˙˙˙˙˙˙
p 
Ş "˘

0˙˙˙˙˙˙˙˙˙Ì
%__inference_signature_wrapper_5389818˘k˘h
˘ 
aŞ^
-
input_1"
input_1˙˙˙˙˙˙˙˙˙
-
input_2"
input_2˙˙˙˙˙˙˙˙˙"/Ş,
*
output_1
output_1˙˙˙˙˙˙˙˙˙