û
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
 "serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8?
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
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ô8*
shared_namedense_3/kernel
s
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel* 
_output_shapes
:
ô8*
dtype0
q
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
j
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes	
:*
dtype0

Adam/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ô8*&
shared_nameAdam/dense_3/kernel/m

)Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/m* 
_output_shapes
:
ô8*
dtype0

Adam/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/m
x
'Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ô8*&
shared_nameAdam/dense_3/kernel/v

)Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/v* 
_output_shapes
:
ô8*
dtype0

Adam/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/v
x
'Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/v*
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
VARIABLE_VALUEdense_3/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense_3/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEAdam/dense_3/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense_3/bias/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_3/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense_3/bias/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
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
ï
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2dense_3/kerneldense_3/bias*
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
GPU2*0J 8 */
f*R(
&__inference_signature_wrapper_15789066
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ú
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenametotal/Read/ReadVariableOpcount/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp)Adam/dense_3/kernel/m/Read/ReadVariableOp'Adam/dense_3/bias/m/Read/ReadVariableOp)Adam/dense_3/kernel/v/Read/ReadVariableOp'Adam/dense_3/bias/v/Read/ReadVariableOpConst*
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
GPU2*0J 8 **
f%R#
!__inference__traced_save_15789890
ñ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenametotalcount	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratedense_3/kerneldense_3/biasAdam/dense_3/kernel/mAdam/dense_3/bias/mAdam/dense_3/kernel/vAdam/dense_3/bias/v*
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
GPU2*0J 8 *-
f(R&
$__inference__traced_restore_15789939ŜÑ

?
cond_false_15789324
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
â
9
cond_true_15789663
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

9
cond_true_15789600
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

9
cond_true_15789720
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
?
Ĉ
M__inference_siamese_model_3_layer_call_and_return_conditional_losses_15789029	
input
input_1
model_3_15789023
model_3_15789025
identity˘model_3/StatefulPartitionedCall
model_3/StatefulPartitionedCallStatefulPartitionedCallinputinput_1model_3_15789023model_3_15789025*
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
GPU2*0J 8 *N
fIRG
E__inference_model_3_layer_call_and_return_conditional_losses_157889782!
model_3/StatefulPartitionedCall
IdentityIdentity(model_3/StatefulPartitionedCall:output:0 ^model_3/StatefulPartitionedCall*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2B
model_3/StatefulPartitionedCallmodel_3/StatefulPartitionedCall:O K
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameinput:OK
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameinput
Ğ
e
,__inference_dropout_3_layer_call_fn_15789760

inputs
identity˘StatefulPartitionedCallá
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
GPU2*0J 8 *P
fKRI
G__inference_dropout_3_layer_call_and_return_conditional_losses_157886202
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

H
,__inference_dropout_3_layer_call_fn_15789765

inputs
identityÉ
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
GPU2*0J 8 *P
fKRI
G__inference_dropout_3_layer_call_and_return_conditional_losses_157886252
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
Ô@

E__inference_model_3_layer_call_and_return_conditional_losses_15788954

inputs
inputs_17
3sequential_3_dense_3_matmul_readvariableop_resource8
4sequential_3_dense_3_biasadd_readvariableop_resource
identity˘+sequential_3/dense_3/BiasAdd/ReadVariableOp˘-sequential_3/dense_3/BiasAdd_1/ReadVariableOp˘*sequential_3/dense_3/MatMul/ReadVariableOp˘,sequential_3/dense_3/MatMul_1/ReadVariableOp
$sequential_3/dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2&
$sequential_3/dropout_3/dropout/Constı
"sequential_3/dropout_3/dropout/MulMulinputs-sequential_3/dropout_3/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2$
"sequential_3/dropout_3/dropout/Mul
$sequential_3/dropout_3/dropout/ShapeShapeinputs*
T0*
_output_shapes
:2&
$sequential_3/dropout_3/dropout/Shapeú
;sequential_3/dropout_3/dropout/random_uniform/RandomUniformRandomUniform-sequential_3/dropout_3/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02=
;sequential_3/dropout_3/dropout/random_uniform/RandomUniform£
-sequential_3/dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2/
-sequential_3/dropout_3/dropout/GreaterEqual/y
+sequential_3/dropout_3/dropout/GreaterEqualGreaterEqualDsequential_3/dropout_3/dropout/random_uniform/RandomUniform:output:06sequential_3/dropout_3/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2-
+sequential_3/dropout_3/dropout/GreaterEqualĊ
#sequential_3/dropout_3/dropout/CastCast/sequential_3/dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#sequential_3/dropout_3/dropout/Cast×
$sequential_3/dropout_3/dropout/Mul_1Mul&sequential_3/dropout_3/dropout/Mul:z:0'sequential_3/dropout_3/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2&
$sequential_3/dropout_3/dropout/Mul_1
5sequential_3/one_hot_encoding_layer_3/PartitionedCallPartitionedCall(sequential_3/dropout_3/dropout/Mul_1:z:0*
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
GPU2*0J 8 *"
fR
__inference_call_1578852027
5sequential_3/one_hot_encoding_layer_3/PartitionedCall
sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2
sequential_3/flatten_3/Constċ
sequential_3/flatten_3/ReshapeReshape>sequential_3/one_hot_encoding_layer_3/PartitionedCall:output:0%sequential_3/flatten_3/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82 
sequential_3/flatten_3/ReshapeÎ
*sequential_3/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02,
*sequential_3/dense_3/MatMul/ReadVariableOpÔ
sequential_3/dense_3/MatMulMatMul'sequential_3/flatten_3/Reshape:output:02sequential_3/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_3/dense_3/MatMulÌ
+sequential_3/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+sequential_3/dense_3/BiasAdd/ReadVariableOpÖ
sequential_3/dense_3/BiasAddBiasAdd%sequential_3/dense_3/MatMul:product:03sequential_3/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_3/dense_3/BiasAdd
&sequential_3/dropout_3/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2(
&sequential_3/dropout_3/dropout_1/ConstÁ
$sequential_3/dropout_3/dropout_1/MulMulinputs_1/sequential_3/dropout_3/dropout_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2&
$sequential_3/dropout_3/dropout_1/Mul
&sequential_3/dropout_3/dropout_1/ShapeShapeinputs_1*
T0*
_output_shapes
:2(
&sequential_3/dropout_3/dropout_1/Shape
=sequential_3/dropout_3/dropout_1/random_uniform/RandomUniformRandomUniform/sequential_3/dropout_3/dropout_1/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02?
=sequential_3/dropout_3/dropout_1/random_uniform/RandomUniform§
/sequential_3/dropout_3/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    21
/sequential_3/dropout_3/dropout_1/GreaterEqual/y£
-sequential_3/dropout_3/dropout_1/GreaterEqualGreaterEqualFsequential_3/dropout_3/dropout_1/random_uniform/RandomUniform:output:08sequential_3/dropout_3/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2/
-sequential_3/dropout_3/dropout_1/GreaterEqualË
%sequential_3/dropout_3/dropout_1/CastCast1sequential_3/dropout_3/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%sequential_3/dropout_3/dropout_1/Castß
&sequential_3/dropout_3/dropout_1/Mul_1Mul(sequential_3/dropout_3/dropout_1/Mul:z:0)sequential_3/dropout_3/dropout_1/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2(
&sequential_3/dropout_3/dropout_1/Mul_1
7sequential_3/one_hot_encoding_layer_3/PartitionedCall_1PartitionedCall*sequential_3/dropout_3/dropout_1/Mul_1:z:0*
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
GPU2*0J 8 *"
fR
__inference_call_1578852029
7sequential_3/one_hot_encoding_layer_3/PartitionedCall_1
sequential_3/flatten_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2 
sequential_3/flatten_3/Const_1í
 sequential_3/flatten_3/Reshape_1Reshape@sequential_3/one_hot_encoding_layer_3/PartitionedCall_1:output:0'sequential_3/flatten_3/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82"
 sequential_3/flatten_3/Reshape_1Ò
,sequential_3/dense_3/MatMul_1/ReadVariableOpReadVariableOp3sequential_3_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02.
,sequential_3/dense_3/MatMul_1/ReadVariableOpÜ
sequential_3/dense_3/MatMul_1MatMul)sequential_3/flatten_3/Reshape_1:output:04sequential_3/dense_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_3/dense_3/MatMul_1?
-sequential_3/dense_3/BiasAdd_1/ReadVariableOpReadVariableOp4sequential_3_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-sequential_3/dense_3/BiasAdd_1/ReadVariableOpŜ
sequential_3/dense_3/BiasAdd_1BiasAdd'sequential_3/dense_3/MatMul_1:product:05sequential_3/dense_3/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
sequential_3/dense_3/BiasAdd_1
 distance_layer_3/PartitionedCallPartitionedCall%sequential_3/dense_3/BiasAdd:output:0'sequential_3/dense_3/BiasAdd_1:output:0*
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
GPU2*0J 8 *"
fR
__inference_call_157885942"
 distance_layer_3/PartitionedCall³
IdentityIdentity)distance_layer_3/PartitionedCall:output:0,^sequential_3/dense_3/BiasAdd/ReadVariableOp.^sequential_3/dense_3/BiasAdd_1/ReadVariableOp+^sequential_3/dense_3/MatMul/ReadVariableOp-^sequential_3/dense_3/MatMul_1/ReadVariableOp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2Z
+sequential_3/dense_3/BiasAdd/ReadVariableOp+sequential_3/dense_3/BiasAdd/ReadVariableOp2^
-sequential_3/dense_3/BiasAdd_1/ReadVariableOp-sequential_3/dense_3/BiasAdd_1/ReadVariableOp2X
*sequential_3/dense_3/MatMul/ReadVariableOp*sequential_3/dense_3/MatMul/ReadVariableOp2\
,sequential_3/dense_3/MatMul_1/ReadVariableOp,sequential_3/dense_3/MatMul_1/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:PL
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
?

*__inference_model_3_layer_call_fn_15789426
inputs_0
inputs_1
unknown
	unknown_0
identity˘StatefulPartitionedCall
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
GPU2*0J 8 *N
fIRG
E__inference_model_3_layer_call_and_return_conditional_losses_157889782
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
î-
£
__inference_call_15789254
input_0
input_1?
;model_3_sequential_3_dense_3_matmul_readvariableop_resource@
<model_3_sequential_3_dense_3_biasadd_readvariableop_resource
identity˘3model_3/sequential_3/dense_3/BiasAdd/ReadVariableOp˘5model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOp˘2model_3/sequential_3/dense_3/MatMul/ReadVariableOp˘4model_3/sequential_3/dense_3/MatMul_1/ReadVariableOp
'model_3/sequential_3/dropout_3/IdentityIdentityinput_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2)
'model_3/sequential_3/dropout_3/Identity?
=model_3/sequential_3/one_hot_encoding_layer_3/PartitionedCallPartitionedCall0model_3/sequential_3/dropout_3/Identity:output:0*
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
GPU2*0J 8 *"
fR
__inference_call_157885202?
=model_3/sequential_3/one_hot_encoding_layer_3/PartitionedCall
$model_3/sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2&
$model_3/sequential_3/flatten_3/Const
&model_3/sequential_3/flatten_3/ReshapeReshapeFmodel_3/sequential_3/one_hot_encoding_layer_3/PartitionedCall:output:0-model_3/sequential_3/flatten_3/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82(
&model_3/sequential_3/flatten_3/Reshapeĉ
2model_3/sequential_3/dense_3/MatMul/ReadVariableOpReadVariableOp;model_3_sequential_3_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype024
2model_3/sequential_3/dense_3/MatMul/ReadVariableOpô
#model_3/sequential_3/dense_3/MatMulMatMul/model_3/sequential_3/flatten_3/Reshape:output:0:model_3/sequential_3/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#model_3/sequential_3/dense_3/MatMulä
3model_3/sequential_3/dense_3/BiasAdd/ReadVariableOpReadVariableOp<model_3_sequential_3_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype025
3model_3/sequential_3/dense_3/BiasAdd/ReadVariableOpö
$model_3/sequential_3/dense_3/BiasAddBiasAdd-model_3/sequential_3/dense_3/MatMul:product:0;model_3/sequential_3/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2&
$model_3/sequential_3/dense_3/BiasAdd
)model_3/sequential_3/dropout_3/Identity_1Identityinput_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2+
)model_3/sequential_3/dropout_3/Identity_1Ğ
?model_3/sequential_3/one_hot_encoding_layer_3/PartitionedCall_1PartitionedCall2model_3/sequential_3/dropout_3/Identity_1:output:0*
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
GPU2*0J 8 *"
fR
__inference_call_157885202A
?model_3/sequential_3/one_hot_encoding_layer_3/PartitionedCall_1Ħ
&model_3/sequential_3/flatten_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2(
&model_3/sequential_3/flatten_3/Const_1
(model_3/sequential_3/flatten_3/Reshape_1ReshapeHmodel_3/sequential_3/one_hot_encoding_layer_3/PartitionedCall_1:output:0/model_3/sequential_3/flatten_3/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82*
(model_3/sequential_3/flatten_3/Reshape_1ê
4model_3/sequential_3/dense_3/MatMul_1/ReadVariableOpReadVariableOp;model_3_sequential_3_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype026
4model_3/sequential_3/dense_3/MatMul_1/ReadVariableOpü
%model_3/sequential_3/dense_3/MatMul_1MatMul1model_3/sequential_3/flatten_3/Reshape_1:output:0<model_3/sequential_3/dense_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%model_3/sequential_3/dense_3/MatMul_1è
5model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOpReadVariableOp<model_3_sequential_3_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype027
5model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOpŝ
&model_3/sequential_3/dense_3/BiasAdd_1BiasAdd/model_3/sequential_3/dense_3/MatMul_1:product:0=model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2(
&model_3/sequential_3/dense_3/BiasAdd_1Ħ
(model_3/distance_layer_3/PartitionedCallPartitionedCall-model_3/sequential_3/dense_3/BiasAdd:output:0/model_3/sequential_3/dense_3/BiasAdd_1:output:0*
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
GPU2*0J 8 *"
fR
__inference_call_157885942*
(model_3/distance_layer_3/PartitionedCallÛ
IdentityIdentity1model_3/distance_layer_3/PartitionedCall:output:04^model_3/sequential_3/dense_3/BiasAdd/ReadVariableOp6^model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOp3^model_3/sequential_3/dense_3/MatMul/ReadVariableOp5^model_3/sequential_3/dense_3/MatMul_1/ReadVariableOp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2j
3model_3/sequential_3/dense_3/BiasAdd/ReadVariableOp3model_3/sequential_3/dense_3/BiasAdd/ReadVariableOp2n
5model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOp5model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOp2h
2model_3/sequential_3/dense_3/MatMul/ReadVariableOp2model_3/sequential_3/dense_3/MatMul/ReadVariableOp2l
4model_3/sequential_3/dense_3/MatMul_1/ReadVariableOp4model_3/sequential_3/dense_3/MatMul_1/ReadVariableOp:Q M
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
½
m
V__inference_one_hot_encoding_layer_3_layer_call_and_return_conditional_losses_15789774
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
ı
c
G__inference_flatten_3_layer_call_and_return_conditional_losses_15789803

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
ĥ
ü
E__inference_model_3_layer_call_and_return_conditional_losses_15788863
	sequence1
	sequence2
sequential_3_15788853
sequential_3_15788855
identity˘$sequential_3/StatefulPartitionedCall˘&sequential_3/StatefulPartitionedCall_1µ
$sequential_3/StatefulPartitionedCallStatefulPartitionedCall	sequence1sequential_3_15788853sequential_3_15788855*
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
GPU2*0J 8 *S
fNRL
J__inference_sequential_3_layer_call_and_return_conditional_losses_157887442&
$sequential_3/StatefulPartitionedCallı
&sequential_3/StatefulPartitionedCall_1StatefulPartitionedCall	sequence2sequential_3_15788853sequential_3_15788855*
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
GPU2*0J 8 *S
fNRL
J__inference_sequential_3_layer_call_and_return_conditional_losses_157887442(
&sequential_3/StatefulPartitionedCall_1Ĉ
 distance_layer_3/PartitionedCallPartitionedCall-sequential_3/StatefulPartitionedCall:output:0/sequential_3/StatefulPartitionedCall_1:output:0*
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
GPU2*0J 8 *W
fRRP
N__inference_distance_layer_3_layer_call_and_return_conditional_losses_157888392"
 distance_layer_3/PartitionedCallÉ
IdentityIdentity)distance_layer_3/PartitionedCall:output:0%^sequential_3/StatefulPartitionedCall'^sequential_3/StatefulPartitionedCall_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall2P
&sequential_3/StatefulPartitionedCall_1&sequential_3/StatefulPartitionedCall_1:S O
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

f
G__inference_dropout_3_layer_call_and_return_conditional_losses_15789750

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
ĥ
ü
E__inference_model_3_layer_call_and_return_conditional_losses_15788849
	sequence1
	sequence2
sequential_3_15788774
sequential_3_15788776
identity˘$sequential_3/StatefulPartitionedCall˘&sequential_3/StatefulPartitionedCall_1µ
$sequential_3/StatefulPartitionedCallStatefulPartitionedCall	sequence1sequential_3_15788774sequential_3_15788776*
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
GPU2*0J 8 *S
fNRL
J__inference_sequential_3_layer_call_and_return_conditional_losses_157887232&
$sequential_3/StatefulPartitionedCallı
&sequential_3/StatefulPartitionedCall_1StatefulPartitionedCall	sequence2sequential_3_15788774sequential_3_15788776*
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
GPU2*0J 8 *S
fNRL
J__inference_sequential_3_layer_call_and_return_conditional_losses_157887232(
&sequential_3/StatefulPartitionedCall_1Ĉ
 distance_layer_3/PartitionedCallPartitionedCall-sequential_3/StatefulPartitionedCall:output:0/sequential_3/StatefulPartitionedCall_1:output:0*
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
GPU2*0J 8 *W
fRRP
N__inference_distance_layer_3_layer_call_and_return_conditional_losses_157888392"
 distance_layer_3/PartitionedCallÉ
IdentityIdentity)distance_layer_3/PartitionedCall:output:0%^sequential_3/StatefulPartitionedCall'^sequential_3/StatefulPartitionedCall_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall2P
&sequential_3/StatefulPartitionedCall_1&sequential_3/StatefulPartitionedCall_1:S O
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
?

*__inference_model_3_layer_call_fn_15789498
inputs_0
inputs_1
unknown
	unknown_0
identity˘StatefulPartitionedCall
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
GPU2*0J 8 *N
fIRG
E__inference_model_3_layer_call_and_return_conditional_losses_157888812
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
í

2__inference_siamese_model_3_layer_call_fn_15789230
input_0
input_1
unknown
	unknown_0
identity˘StatefulPartitionedCall
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
GPU2*0J 8 *V
fQRO
M__inference_siamese_model_3_layer_call_and_return_conditional_losses_157890292
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
"
9
__inference_call_15789738
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

LogicalNotı
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
 *&
else_branchR
cond_false_15789721*"
output_shapes
:˙˙˙˙˙˙˙˙˙*%
then_branchR
cond_true_157897202
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

0
__inference_call_15788520
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

0
__inference_call_15789788
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
Ĝ
0
__inference_call_15789797
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
Ċ
Ü
J__inference_sequential_3_layer_call_and_return_conditional_losses_15788696
input_4
dense_3_15788690
dense_3_15788692
identity˘dense_3/StatefulPartitionedCall˘!dropout_3/StatefulPartitionedCallö
!dropout_3/StatefulPartitionedCallStatefulPartitionedCallinput_4*
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
GPU2*0J 8 *P
fKRI
G__inference_dropout_3_layer_call_and_return_conditional_losses_157886202#
!dropout_3/StatefulPartitionedCall²
(one_hot_encoding_layer_3/PartitionedCallPartitionedCall*dropout_3/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *_
fZRX
V__inference_one_hot_encoding_layer_3_layer_call_and_return_conditional_losses_157886472*
(one_hot_encoding_layer_3/PartitionedCall
flatten_3/PartitionedCallPartitionedCall1one_hot_encoding_layer_3/PartitionedCall:output:0*
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
GPU2*0J 8 *P
fKRI
G__inference_flatten_3_layer_call_and_return_conditional_losses_157886612
flatten_3/PartitionedCallµ
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_3_15788690dense_3_15788692*
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
GPU2*0J 8 *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_157886792!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_3/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall:Q M
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_4
ı
c
G__inference_flatten_3_layer_call_and_return_conditional_losses_15788661

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
§
H
,__inference_flatten_3_layer_call_fn_15789808

inputs
identityÉ
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
GPU2*0J 8 *P
fKRI
G__inference_flatten_3_layer_call_and_return_conditional_losses_157886612
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
"
9
__inference_call_15788594
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

LogicalNotı
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
 *&
else_branchR
cond_false_15788577*"
output_shapes
:˙˙˙˙˙˙˙˙˙*%
then_branchR
cond_true_157885762
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
ĉ-
Ħ
__inference_call_15788597	
input
input_1?
;model_3_sequential_3_dense_3_matmul_readvariableop_resource@
<model_3_sequential_3_dense_3_biasadd_readvariableop_resource
identity˘3model_3/sequential_3/dense_3/BiasAdd/ReadVariableOp˘5model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOp˘2model_3/sequential_3/dense_3/MatMul/ReadVariableOp˘4model_3/sequential_3/dense_3/MatMul_1/ReadVariableOp
'model_3/sequential_3/dropout_3/IdentityIdentityinput*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2)
'model_3/sequential_3/dropout_3/Identity?
=model_3/sequential_3/one_hot_encoding_layer_3/PartitionedCallPartitionedCall0model_3/sequential_3/dropout_3/Identity:output:0*
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
GPU2*0J 8 *"
fR
__inference_call_157885202?
=model_3/sequential_3/one_hot_encoding_layer_3/PartitionedCall
$model_3/sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2&
$model_3/sequential_3/flatten_3/Const
&model_3/sequential_3/flatten_3/ReshapeReshapeFmodel_3/sequential_3/one_hot_encoding_layer_3/PartitionedCall:output:0-model_3/sequential_3/flatten_3/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82(
&model_3/sequential_3/flatten_3/Reshapeĉ
2model_3/sequential_3/dense_3/MatMul/ReadVariableOpReadVariableOp;model_3_sequential_3_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype024
2model_3/sequential_3/dense_3/MatMul/ReadVariableOpô
#model_3/sequential_3/dense_3/MatMulMatMul/model_3/sequential_3/flatten_3/Reshape:output:0:model_3/sequential_3/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#model_3/sequential_3/dense_3/MatMulä
3model_3/sequential_3/dense_3/BiasAdd/ReadVariableOpReadVariableOp<model_3_sequential_3_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype025
3model_3/sequential_3/dense_3/BiasAdd/ReadVariableOpö
$model_3/sequential_3/dense_3/BiasAddBiasAdd-model_3/sequential_3/dense_3/MatMul:product:0;model_3/sequential_3/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2&
$model_3/sequential_3/dense_3/BiasAdd
)model_3/sequential_3/dropout_3/Identity_1Identityinput_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2+
)model_3/sequential_3/dropout_3/Identity_1Ğ
?model_3/sequential_3/one_hot_encoding_layer_3/PartitionedCall_1PartitionedCall2model_3/sequential_3/dropout_3/Identity_1:output:0*
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
GPU2*0J 8 *"
fR
__inference_call_157885202A
?model_3/sequential_3/one_hot_encoding_layer_3/PartitionedCall_1Ħ
&model_3/sequential_3/flatten_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2(
&model_3/sequential_3/flatten_3/Const_1
(model_3/sequential_3/flatten_3/Reshape_1ReshapeHmodel_3/sequential_3/one_hot_encoding_layer_3/PartitionedCall_1:output:0/model_3/sequential_3/flatten_3/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82*
(model_3/sequential_3/flatten_3/Reshape_1ê
4model_3/sequential_3/dense_3/MatMul_1/ReadVariableOpReadVariableOp;model_3_sequential_3_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype026
4model_3/sequential_3/dense_3/MatMul_1/ReadVariableOpü
%model_3/sequential_3/dense_3/MatMul_1MatMul1model_3/sequential_3/flatten_3/Reshape_1:output:0<model_3/sequential_3/dense_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%model_3/sequential_3/dense_3/MatMul_1è
5model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOpReadVariableOp<model_3_sequential_3_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype027
5model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOpŝ
&model_3/sequential_3/dense_3/BiasAdd_1BiasAdd/model_3/sequential_3/dense_3/MatMul_1:product:0=model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2(
&model_3/sequential_3/dense_3/BiasAdd_1Ħ
(model_3/distance_layer_3/PartitionedCallPartitionedCall-model_3/sequential_3/dense_3/BiasAdd:output:0/model_3/sequential_3/dense_3/BiasAdd_1:output:0*
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
GPU2*0J 8 *"
fR
__inference_call_157885942*
(model_3/distance_layer_3/PartitionedCallÛ
IdentityIdentity1model_3/distance_layer_3/PartitionedCall:output:04^model_3/sequential_3/dense_3/BiasAdd/ReadVariableOp6^model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOp3^model_3/sequential_3/dense_3/MatMul/ReadVariableOp5^model_3/sequential_3/dense_3/MatMul_1/ReadVariableOp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2j
3model_3/sequential_3/dense_3/BiasAdd/ReadVariableOp3model_3/sequential_3/dense_3/BiasAdd/ReadVariableOp2n
5model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOp5model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOp2h
2model_3/sequential_3/dense_3/MatMul/ReadVariableOp2model_3/sequential_3/dense_3/MatMul/ReadVariableOp2l
4model_3/sequential_3/dense_3/MatMul_1/ReadVariableOp4model_3/sequential_3/dense_3/MatMul_1/ReadVariableOp:O K
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameinput:OK
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameinput
í

2__inference_siamese_model_3_layer_call_fn_15789148
input_1
input_2
unknown
	unknown_0
identity˘StatefulPartitionedCall
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
GPU2*0J 8 *V
fQRO
M__inference_siamese_model_3_layer_call_and_return_conditional_losses_157890292
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
¨
ĝ
E__inference_model_3_layer_call_and_return_conditional_losses_15788881

inputs
inputs_1
sequential_3_15788871
sequential_3_15788873
identity˘$sequential_3/StatefulPartitionedCall˘&sequential_3/StatefulPartitionedCall_1²
$sequential_3/StatefulPartitionedCallStatefulPartitionedCallinputssequential_3_15788871sequential_3_15788873*
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
GPU2*0J 8 *S
fNRL
J__inference_sequential_3_layer_call_and_return_conditional_losses_157887232&
$sequential_3/StatefulPartitionedCall¸
&sequential_3/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1sequential_3_15788871sequential_3_15788873*
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
GPU2*0J 8 *S
fNRL
J__inference_sequential_3_layer_call_and_return_conditional_losses_157887232(
&sequential_3/StatefulPartitionedCall_1Ĉ
 distance_layer_3/PartitionedCallPartitionedCall-sequential_3/StatefulPartitionedCall:output:0/sequential_3/StatefulPartitionedCall_1:output:0*
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
GPU2*0J 8 *W
fRRP
N__inference_distance_layer_3_layer_call_and_return_conditional_losses_157888392"
 distance_layer_3/PartitionedCallÉ
IdentityIdentity)distance_layer_3/PartitionedCall:output:0%^sequential_3/StatefulPartitionedCall'^sequential_3/StatefulPartitionedCall_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall2P
&sequential_3/StatefulPartitionedCall_1&sequential_3/StatefulPartitionedCall_1:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:PL
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
é

*__inference_model_3_layer_call_fn_15788912
	sequence1
	sequence2
unknown
	unknown_0
identity˘StatefulPartitionedCall
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
GPU2*0J 8 *N
fIRG
E__inference_model_3_layer_call_and_return_conditional_losses_157889052
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
é

*__inference_model_3_layer_call_fn_15788888
	sequence1
	sequence2
unknown
	unknown_0
identity˘StatefulPartitionedCall
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
GPU2*0J 8 *N
fIRG
E__inference_model_3_layer_call_and_return_conditional_losses_157888812
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
˘.
×
M__inference_siamese_model_3_layer_call_and_return_conditional_losses_15789128
input_1
input_2?
;model_3_sequential_3_dense_3_matmul_readvariableop_resource@
<model_3_sequential_3_dense_3_biasadd_readvariableop_resource
identity˘3model_3/sequential_3/dense_3/BiasAdd/ReadVariableOp˘5model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOp˘2model_3/sequential_3/dense_3/MatMul/ReadVariableOp˘4model_3/sequential_3/dense_3/MatMul_1/ReadVariableOp
'model_3/sequential_3/dropout_3/IdentityIdentityinput_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2)
'model_3/sequential_3/dropout_3/Identity?
=model_3/sequential_3/one_hot_encoding_layer_3/PartitionedCallPartitionedCall0model_3/sequential_3/dropout_3/Identity:output:0*
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
GPU2*0J 8 *"
fR
__inference_call_157885202?
=model_3/sequential_3/one_hot_encoding_layer_3/PartitionedCall
$model_3/sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2&
$model_3/sequential_3/flatten_3/Const
&model_3/sequential_3/flatten_3/ReshapeReshapeFmodel_3/sequential_3/one_hot_encoding_layer_3/PartitionedCall:output:0-model_3/sequential_3/flatten_3/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82(
&model_3/sequential_3/flatten_3/Reshapeĉ
2model_3/sequential_3/dense_3/MatMul/ReadVariableOpReadVariableOp;model_3_sequential_3_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype024
2model_3/sequential_3/dense_3/MatMul/ReadVariableOpô
#model_3/sequential_3/dense_3/MatMulMatMul/model_3/sequential_3/flatten_3/Reshape:output:0:model_3/sequential_3/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#model_3/sequential_3/dense_3/MatMulä
3model_3/sequential_3/dense_3/BiasAdd/ReadVariableOpReadVariableOp<model_3_sequential_3_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype025
3model_3/sequential_3/dense_3/BiasAdd/ReadVariableOpö
$model_3/sequential_3/dense_3/BiasAddBiasAdd-model_3/sequential_3/dense_3/MatMul:product:0;model_3/sequential_3/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2&
$model_3/sequential_3/dense_3/BiasAdd
)model_3/sequential_3/dropout_3/Identity_1Identityinput_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2+
)model_3/sequential_3/dropout_3/Identity_1Ğ
?model_3/sequential_3/one_hot_encoding_layer_3/PartitionedCall_1PartitionedCall2model_3/sequential_3/dropout_3/Identity_1:output:0*
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
GPU2*0J 8 *"
fR
__inference_call_157885202A
?model_3/sequential_3/one_hot_encoding_layer_3/PartitionedCall_1Ħ
&model_3/sequential_3/flatten_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2(
&model_3/sequential_3/flatten_3/Const_1
(model_3/sequential_3/flatten_3/Reshape_1ReshapeHmodel_3/sequential_3/one_hot_encoding_layer_3/PartitionedCall_1:output:0/model_3/sequential_3/flatten_3/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82*
(model_3/sequential_3/flatten_3/Reshape_1ê
4model_3/sequential_3/dense_3/MatMul_1/ReadVariableOpReadVariableOp;model_3_sequential_3_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype026
4model_3/sequential_3/dense_3/MatMul_1/ReadVariableOpü
%model_3/sequential_3/dense_3/MatMul_1MatMul1model_3/sequential_3/flatten_3/Reshape_1:output:0<model_3/sequential_3/dense_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%model_3/sequential_3/dense_3/MatMul_1è
5model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOpReadVariableOp<model_3_sequential_3_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype027
5model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOpŝ
&model_3/sequential_3/dense_3/BiasAdd_1BiasAdd/model_3/sequential_3/dense_3/MatMul_1:product:0=model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2(
&model_3/sequential_3/dense_3/BiasAdd_1Ħ
(model_3/distance_layer_3/PartitionedCallPartitionedCall-model_3/sequential_3/dense_3/BiasAdd:output:0/model_3/sequential_3/dense_3/BiasAdd_1:output:0*
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
GPU2*0J 8 *"
fR
__inference_call_157885942*
(model_3/distance_layer_3/PartitionedCallÛ
IdentityIdentity1model_3/distance_layer_3/PartitionedCall:output:04^model_3/sequential_3/dense_3/BiasAdd/ReadVariableOp6^model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOp3^model_3/sequential_3/dense_3/MatMul/ReadVariableOp5^model_3/sequential_3/dense_3/MatMul_1/ReadVariableOp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2j
3model_3/sequential_3/dense_3/BiasAdd/ReadVariableOp3model_3/sequential_3/dense_3/BiasAdd/ReadVariableOp2n
5model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOp5model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOp2h
2model_3/sequential_3/dense_3/MatMul/ReadVariableOp2model_3/sequential_3/dense_3/MatMul/ReadVariableOp2l
4model_3/sequential_3/dense_3/MatMul_1/ReadVariableOp4model_3/sequential_3/dense_3/MatMul_1/ReadVariableOp:Q M
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
˘.
×
M__inference_siamese_model_3_layer_call_and_return_conditional_losses_15789210
input_0
input_1?
;model_3_sequential_3_dense_3_matmul_readvariableop_resource@
<model_3_sequential_3_dense_3_biasadd_readvariableop_resource
identity˘3model_3/sequential_3/dense_3/BiasAdd/ReadVariableOp˘5model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOp˘2model_3/sequential_3/dense_3/MatMul/ReadVariableOp˘4model_3/sequential_3/dense_3/MatMul_1/ReadVariableOp
'model_3/sequential_3/dropout_3/IdentityIdentityinput_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2)
'model_3/sequential_3/dropout_3/Identity?
=model_3/sequential_3/one_hot_encoding_layer_3/PartitionedCallPartitionedCall0model_3/sequential_3/dropout_3/Identity:output:0*
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
GPU2*0J 8 *"
fR
__inference_call_157885202?
=model_3/sequential_3/one_hot_encoding_layer_3/PartitionedCall
$model_3/sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2&
$model_3/sequential_3/flatten_3/Const
&model_3/sequential_3/flatten_3/ReshapeReshapeFmodel_3/sequential_3/one_hot_encoding_layer_3/PartitionedCall:output:0-model_3/sequential_3/flatten_3/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82(
&model_3/sequential_3/flatten_3/Reshapeĉ
2model_3/sequential_3/dense_3/MatMul/ReadVariableOpReadVariableOp;model_3_sequential_3_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype024
2model_3/sequential_3/dense_3/MatMul/ReadVariableOpô
#model_3/sequential_3/dense_3/MatMulMatMul/model_3/sequential_3/flatten_3/Reshape:output:0:model_3/sequential_3/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#model_3/sequential_3/dense_3/MatMulä
3model_3/sequential_3/dense_3/BiasAdd/ReadVariableOpReadVariableOp<model_3_sequential_3_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype025
3model_3/sequential_3/dense_3/BiasAdd/ReadVariableOpö
$model_3/sequential_3/dense_3/BiasAddBiasAdd-model_3/sequential_3/dense_3/MatMul:product:0;model_3/sequential_3/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2&
$model_3/sequential_3/dense_3/BiasAdd
)model_3/sequential_3/dropout_3/Identity_1Identityinput_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2+
)model_3/sequential_3/dropout_3/Identity_1Ğ
?model_3/sequential_3/one_hot_encoding_layer_3/PartitionedCall_1PartitionedCall2model_3/sequential_3/dropout_3/Identity_1:output:0*
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
GPU2*0J 8 *"
fR
__inference_call_157885202A
?model_3/sequential_3/one_hot_encoding_layer_3/PartitionedCall_1Ħ
&model_3/sequential_3/flatten_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2(
&model_3/sequential_3/flatten_3/Const_1
(model_3/sequential_3/flatten_3/Reshape_1ReshapeHmodel_3/sequential_3/one_hot_encoding_layer_3/PartitionedCall_1:output:0/model_3/sequential_3/flatten_3/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82*
(model_3/sequential_3/flatten_3/Reshape_1ê
4model_3/sequential_3/dense_3/MatMul_1/ReadVariableOpReadVariableOp;model_3_sequential_3_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype026
4model_3/sequential_3/dense_3/MatMul_1/ReadVariableOpü
%model_3/sequential_3/dense_3/MatMul_1MatMul1model_3/sequential_3/flatten_3/Reshape_1:output:0<model_3/sequential_3/dense_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%model_3/sequential_3/dense_3/MatMul_1è
5model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOpReadVariableOp<model_3_sequential_3_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype027
5model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOpŝ
&model_3/sequential_3/dense_3/BiasAdd_1BiasAdd/model_3/sequential_3/dense_3/MatMul_1:product:0=model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2(
&model_3/sequential_3/dense_3/BiasAdd_1Ħ
(model_3/distance_layer_3/PartitionedCallPartitionedCall-model_3/sequential_3/dense_3/BiasAdd:output:0/model_3/sequential_3/dense_3/BiasAdd_1:output:0*
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
GPU2*0J 8 *"
fR
__inference_call_157885942*
(model_3/distance_layer_3/PartitionedCallÛ
IdentityIdentity1model_3/distance_layer_3/PartitionedCall:output:04^model_3/sequential_3/dense_3/BiasAdd/ReadVariableOp6^model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOp3^model_3/sequential_3/dense_3/MatMul/ReadVariableOp5^model_3/sequential_3/dense_3/MatMul_1/ReadVariableOp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2j
3model_3/sequential_3/dense_3/BiasAdd/ReadVariableOp3model_3/sequential_3/dense_3/BiasAdd/ReadVariableOp2n
5model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOp5model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOp2h
2model_3/sequential_3/dense_3/MatMul/ReadVariableOp2model_3/sequential_3/dense_3/MatMul/ReadVariableOp2l
4model_3/sequential_3/dense_3/MatMul_1/ReadVariableOp4model_3/sequential_3/dense_3/MatMul_1/ReadVariableOp:Q M
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

9
cond_true_15788576
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

f
G__inference_dropout_3_layer_call_and_return_conditional_losses_15788620

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
öG
×
M__inference_siamese_model_3_layer_call_and_return_conditional_losses_15789186
input_0
input_1?
;model_3_sequential_3_dense_3_matmul_readvariableop_resource@
<model_3_sequential_3_dense_3_biasadd_readvariableop_resource
identity˘3model_3/sequential_3/dense_3/BiasAdd/ReadVariableOp˘5model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOp˘2model_3/sequential_3/dense_3/MatMul/ReadVariableOp˘4model_3/sequential_3/dense_3/MatMul_1/ReadVariableOpĦ
,model_3/sequential_3/dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2.
,model_3/sequential_3/dropout_3/dropout/ConstÒ
*model_3/sequential_3/dropout_3/dropout/MulMulinput_05model_3/sequential_3/dropout_3/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2,
*model_3/sequential_3/dropout_3/dropout/Mul
,model_3/sequential_3/dropout_3/dropout/ShapeShapeinput_0*
T0*
_output_shapes
:2.
,model_3/sequential_3/dropout_3/dropout/Shape
Cmodel_3/sequential_3/dropout_3/dropout/random_uniform/RandomUniformRandomUniform5model_3/sequential_3/dropout_3/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02E
Cmodel_3/sequential_3/dropout_3/dropout/random_uniform/RandomUniform³
5model_3/sequential_3/dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5model_3/sequential_3/dropout_3/dropout/GreaterEqual/yğ
3model_3/sequential_3/dropout_3/dropout/GreaterEqualGreaterEqualLmodel_3/sequential_3/dropout_3/dropout/random_uniform/RandomUniform:output:0>model_3/sequential_3/dropout_3/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙25
3model_3/sequential_3/dropout_3/dropout/GreaterEqualŬ
+model_3/sequential_3/dropout_3/dropout/CastCast7model_3/sequential_3/dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2-
+model_3/sequential_3/dropout_3/dropout/Cast÷
,model_3/sequential_3/dropout_3/dropout/Mul_1Mul.model_3/sequential_3/dropout_3/dropout/Mul:z:0/model_3/sequential_3/dropout_3/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2.
,model_3/sequential_3/dropout_3/dropout/Mul_1?
=model_3/sequential_3/one_hot_encoding_layer_3/PartitionedCallPartitionedCall0model_3/sequential_3/dropout_3/dropout/Mul_1:z:0*
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
GPU2*0J 8 *"
fR
__inference_call_157885202?
=model_3/sequential_3/one_hot_encoding_layer_3/PartitionedCall
$model_3/sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2&
$model_3/sequential_3/flatten_3/Const
&model_3/sequential_3/flatten_3/ReshapeReshapeFmodel_3/sequential_3/one_hot_encoding_layer_3/PartitionedCall:output:0-model_3/sequential_3/flatten_3/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82(
&model_3/sequential_3/flatten_3/Reshapeĉ
2model_3/sequential_3/dense_3/MatMul/ReadVariableOpReadVariableOp;model_3_sequential_3_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype024
2model_3/sequential_3/dense_3/MatMul/ReadVariableOpô
#model_3/sequential_3/dense_3/MatMulMatMul/model_3/sequential_3/flatten_3/Reshape:output:0:model_3/sequential_3/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#model_3/sequential_3/dense_3/MatMulä
3model_3/sequential_3/dense_3/BiasAdd/ReadVariableOpReadVariableOp<model_3_sequential_3_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype025
3model_3/sequential_3/dense_3/BiasAdd/ReadVariableOpö
$model_3/sequential_3/dense_3/BiasAddBiasAdd-model_3/sequential_3/dense_3/MatMul:product:0;model_3/sequential_3/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2&
$model_3/sequential_3/dense_3/BiasAdd?
.model_3/sequential_3/dropout_3/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?20
.model_3/sequential_3/dropout_3/dropout_1/ConstĜ
,model_3/sequential_3/dropout_3/dropout_1/MulMulinput_17model_3/sequential_3/dropout_3/dropout_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2.
,model_3/sequential_3/dropout_3/dropout_1/Mul
.model_3/sequential_3/dropout_3/dropout_1/ShapeShapeinput_1*
T0*
_output_shapes
:20
.model_3/sequential_3/dropout_3/dropout_1/Shape
Emodel_3/sequential_3/dropout_3/dropout_1/random_uniform/RandomUniformRandomUniform7model_3/sequential_3/dropout_3/dropout_1/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02G
Emodel_3/sequential_3/dropout_3/dropout_1/random_uniform/RandomUniform·
7model_3/sequential_3/dropout_3/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    29
7model_3/sequential_3/dropout_3/dropout_1/GreaterEqual/y?
5model_3/sequential_3/dropout_3/dropout_1/GreaterEqualGreaterEqualNmodel_3/sequential_3/dropout_3/dropout_1/random_uniform/RandomUniform:output:0@model_3/sequential_3/dropout_3/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙27
5model_3/sequential_3/dropout_3/dropout_1/GreaterEqual?
-model_3/sequential_3/dropout_3/dropout_1/CastCast9model_3/sequential_3/dropout_3/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2/
-model_3/sequential_3/dropout_3/dropout_1/Cast˙
.model_3/sequential_3/dropout_3/dropout_1/Mul_1Mul0model_3/sequential_3/dropout_3/dropout_1/Mul:z:01model_3/sequential_3/dropout_3/dropout_1/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙20
.model_3/sequential_3/dropout_3/dropout_1/Mul_1Ğ
?model_3/sequential_3/one_hot_encoding_layer_3/PartitionedCall_1PartitionedCall2model_3/sequential_3/dropout_3/dropout_1/Mul_1:z:0*
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
GPU2*0J 8 *"
fR
__inference_call_157885202A
?model_3/sequential_3/one_hot_encoding_layer_3/PartitionedCall_1Ħ
&model_3/sequential_3/flatten_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2(
&model_3/sequential_3/flatten_3/Const_1
(model_3/sequential_3/flatten_3/Reshape_1ReshapeHmodel_3/sequential_3/one_hot_encoding_layer_3/PartitionedCall_1:output:0/model_3/sequential_3/flatten_3/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82*
(model_3/sequential_3/flatten_3/Reshape_1ê
4model_3/sequential_3/dense_3/MatMul_1/ReadVariableOpReadVariableOp;model_3_sequential_3_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype026
4model_3/sequential_3/dense_3/MatMul_1/ReadVariableOpü
%model_3/sequential_3/dense_3/MatMul_1MatMul1model_3/sequential_3/flatten_3/Reshape_1:output:0<model_3/sequential_3/dense_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%model_3/sequential_3/dense_3/MatMul_1è
5model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOpReadVariableOp<model_3_sequential_3_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype027
5model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOpŝ
&model_3/sequential_3/dense_3/BiasAdd_1BiasAdd/model_3/sequential_3/dense_3/MatMul_1:product:0=model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2(
&model_3/sequential_3/dense_3/BiasAdd_1Ħ
(model_3/distance_layer_3/PartitionedCallPartitionedCall-model_3/sequential_3/dense_3/BiasAdd:output:0/model_3/sequential_3/dense_3/BiasAdd_1:output:0*
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
GPU2*0J 8 *"
fR
__inference_call_157885942*
(model_3/distance_layer_3/PartitionedCallÛ
IdentityIdentity1model_3/distance_layer_3/PartitionedCall:output:04^model_3/sequential_3/dense_3/BiasAdd/ReadVariableOp6^model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOp3^model_3/sequential_3/dense_3/MatMul/ReadVariableOp5^model_3/sequential_3/dense_3/MatMul_1/ReadVariableOp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2j
3model_3/sequential_3/dense_3/BiasAdd/ReadVariableOp3model_3/sequential_3/dense_3/BiasAdd/ReadVariableOp2n
5model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOp5model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOp2h
2model_3/sequential_3/dense_3/MatMul/ReadVariableOp2model_3/sequential_3/dense_3/MatMul/ReadVariableOp2l
4model_3/sequential_3/dense_3/MatMul_1/ReadVariableOp4model_3/sequential_3/dense_3/MatMul_1/ReadVariableOp:Q M
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
	
Ŝ
E__inference_dense_3_layer_call_and_return_conditional_losses_15788679

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

·
J__inference_sequential_3_layer_call_and_return_conditional_losses_15788744

inputs
dense_3_15788738
dense_3_15788740
identity˘dense_3/StatefulPartitionedCallŬ
dropout_3/PartitionedCallPartitionedCallinputs*
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
GPU2*0J 8 *P
fKRI
G__inference_dropout_3_layer_call_and_return_conditional_losses_157886252
dropout_3/PartitionedCallŞ
(one_hot_encoding_layer_3/PartitionedCallPartitionedCall"dropout_3/PartitionedCall:output:0*
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
GPU2*0J 8 *_
fZRX
V__inference_one_hot_encoding_layer_3_layer_call_and_return_conditional_losses_157886472*
(one_hot_encoding_layer_3/PartitionedCall
flatten_3/PartitionedCallPartitionedCall1one_hot_encoding_layer_3/PartitionedCall:output:0*
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
GPU2*0J 8 *P
fKRI
G__inference_flatten_3_layer_call_and_return_conditional_losses_157886612
flatten_3/PartitionedCallµ
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_3_15788738dense_3_15788740*
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
GPU2*0J 8 *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_157886792!
dense_3/StatefulPartitionedCall
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_3/StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ĥ
?
cond_false_15788822
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
ċ

*__inference_dense_3_layer_call_fn_15789827

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallù
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
GPU2*0J 8 *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_157886792
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
?

/__inference_sequential_3_layer_call_fn_15789561

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallŝ
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
GPU2*0J 8 *S
fNRL
J__inference_sequential_3_layer_call_and_return_conditional_losses_157887442
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
Î
e
G__inference_dropout_3_layer_call_and_return_conditional_losses_15788625

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
ż 
9
__inference_call_15789341
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

LogicalNotİ
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
 *&
else_branchR
cond_false_15789324*
output_shapes	
:*%
then_branchR
cond_true_157893232
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
í

2__inference_siamese_model_3_layer_call_fn_15789138
input_1
input_2
unknown
	unknown_0
identity˘StatefulPartitionedCall
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
GPU2*0J 8 *V
fQRO
M__inference_siamese_model_3_layer_call_and_return_conditional_losses_157890292
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
Ĥ
?
cond_false_15789721
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
Ĝ
0
__inference_call_15789267
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
?)

E__inference_model_3_layer_call_and_return_conditional_losses_15788978

inputs
inputs_17
3sequential_3_dense_3_matmul_readvariableop_resource8
4sequential_3_dense_3_biasadd_readvariableop_resource
identity˘+sequential_3/dense_3/BiasAdd/ReadVariableOp˘-sequential_3/dense_3/BiasAdd_1/ReadVariableOp˘*sequential_3/dense_3/MatMul/ReadVariableOp˘,sequential_3/dense_3/MatMul_1/ReadVariableOp
sequential_3/dropout_3/IdentityIdentityinputs*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2!
sequential_3/dropout_3/Identity
5sequential_3/one_hot_encoding_layer_3/PartitionedCallPartitionedCall(sequential_3/dropout_3/Identity:output:0*
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
GPU2*0J 8 *"
fR
__inference_call_1578852027
5sequential_3/one_hot_encoding_layer_3/PartitionedCall
sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2
sequential_3/flatten_3/Constċ
sequential_3/flatten_3/ReshapeReshape>sequential_3/one_hot_encoding_layer_3/PartitionedCall:output:0%sequential_3/flatten_3/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82 
sequential_3/flatten_3/ReshapeÎ
*sequential_3/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02,
*sequential_3/dense_3/MatMul/ReadVariableOpÔ
sequential_3/dense_3/MatMulMatMul'sequential_3/flatten_3/Reshape:output:02sequential_3/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_3/dense_3/MatMulÌ
+sequential_3/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+sequential_3/dense_3/BiasAdd/ReadVariableOpÖ
sequential_3/dense_3/BiasAddBiasAdd%sequential_3/dense_3/MatMul:product:03sequential_3/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_3/dense_3/BiasAdd
!sequential_3/dropout_3/Identity_1Identityinputs_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2#
!sequential_3/dropout_3/Identity_1
7sequential_3/one_hot_encoding_layer_3/PartitionedCall_1PartitionedCall*sequential_3/dropout_3/Identity_1:output:0*
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
GPU2*0J 8 *"
fR
__inference_call_1578852029
7sequential_3/one_hot_encoding_layer_3/PartitionedCall_1
sequential_3/flatten_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2 
sequential_3/flatten_3/Const_1í
 sequential_3/flatten_3/Reshape_1Reshape@sequential_3/one_hot_encoding_layer_3/PartitionedCall_1:output:0'sequential_3/flatten_3/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82"
 sequential_3/flatten_3/Reshape_1Ò
,sequential_3/dense_3/MatMul_1/ReadVariableOpReadVariableOp3sequential_3_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02.
,sequential_3/dense_3/MatMul_1/ReadVariableOpÜ
sequential_3/dense_3/MatMul_1MatMul)sequential_3/flatten_3/Reshape_1:output:04sequential_3/dense_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_3/dense_3/MatMul_1?
-sequential_3/dense_3/BiasAdd_1/ReadVariableOpReadVariableOp4sequential_3_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-sequential_3/dense_3/BiasAdd_1/ReadVariableOpŜ
sequential_3/dense_3/BiasAdd_1BiasAdd'sequential_3/dense_3/MatMul_1:product:05sequential_3/dense_3/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
sequential_3/dense_3/BiasAdd_1
 distance_layer_3/PartitionedCallPartitionedCall%sequential_3/dense_3/BiasAdd:output:0'sequential_3/dense_3/BiasAdd_1:output:0*
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
GPU2*0J 8 *"
fR
__inference_call_157885942"
 distance_layer_3/PartitionedCall³
IdentityIdentity)distance_layer_3/PartitionedCall:output:0,^sequential_3/dense_3/BiasAdd/ReadVariableOp.^sequential_3/dense_3/BiasAdd_1/ReadVariableOp+^sequential_3/dense_3/MatMul/ReadVariableOp-^sequential_3/dense_3/MatMul_1/ReadVariableOp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2Z
+sequential_3/dense_3/BiasAdd/ReadVariableOp+sequential_3/dense_3/BiasAdd/ReadVariableOp2^
-sequential_3/dense_3/BiasAdd_1/ReadVariableOp-sequential_3/dense_3/BiasAdd_1/ReadVariableOp2X
*sequential_3/dense_3/MatMul/ReadVariableOp*sequential_3/dense_3/MatMul/ReadVariableOp2\
,sequential_3/dense_3/MatMul_1/ReadVariableOp,sequential_3/dense_3/MatMul_1/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:PL
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ú
ĥ
#__inference__wrapped_model_15788604
input_1
input_2
siamese_model_3_15788598
siamese_model_3_15788600
identity˘'siamese_model_3/StatefulPartitionedCall
'siamese_model_3/StatefulPartitionedCallStatefulPartitionedCallinput_1input_2siamese_model_3_15788598siamese_model_3_15788600*
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
GPU2*0J 8 *"
fR
__inference_call_157885972)
'siamese_model_3/StatefulPartitionedCallŞ
IdentityIdentity0siamese_model_3/StatefulPartitionedCall:output:0(^siamese_model_3/StatefulPartitionedCall*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2R
'siamese_model_3/StatefulPartitionedCall'siamese_model_3/StatefulPartitionedCall:Q M
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
Ô"
n
N__inference_distance_layer_3_layer_call_and_return_conditional_losses_15789618
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

LogicalNotı
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
 *&
else_branchR
cond_false_15789601*"
output_shapes
:˙˙˙˙˙˙˙˙˙*%
then_branchR
cond_true_157896002
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
?

*__inference_model_3_layer_call_fn_15789508
inputs_0
inputs_1
unknown
	unknown_0
identity˘StatefulPartitionedCall
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
GPU2*0J 8 *N
fIRG
E__inference_model_3_layer_call_and_return_conditional_losses_157889052
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
	
Ŝ
E__inference_dense_3_layer_call_and_return_conditional_losses_15789818

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
Î
e
G__inference_dropout_3_layer_call_and_return_conditional_losses_15789755

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
ĥ
R
;__inference_one_hot_encoding_layer_3_layer_call_fn_15789779
x
identity×
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
GPU2*0J 8 *_
fZRX
V__inference_one_hot_encoding_layer_3_layer_call_and_return_conditional_losses_157886472
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

¸
J__inference_sequential_3_layer_call_and_return_conditional_losses_15788708
input_4
dense_3_15788702
dense_3_15788704
identity˘dense_3/StatefulPartitionedCallŜ
dropout_3/PartitionedCallPartitionedCallinput_4*
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
GPU2*0J 8 *P
fKRI
G__inference_dropout_3_layer_call_and_return_conditional_losses_157886252
dropout_3/PartitionedCallŞ
(one_hot_encoding_layer_3/PartitionedCallPartitionedCall"dropout_3/PartitionedCall:output:0*
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
GPU2*0J 8 *_
fZRX
V__inference_one_hot_encoding_layer_3_layer_call_and_return_conditional_losses_157886472*
(one_hot_encoding_layer_3/PartitionedCall
flatten_3/PartitionedCallPartitionedCall1one_hot_encoding_layer_3/PartitionedCall:output:0*
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
GPU2*0J 8 *P
fKRI
G__inference_flatten_3_layer_call_and_return_conditional_losses_157886612
flatten_3/PartitionedCallµ
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_3_15788702dense_3_15788704*
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
GPU2*0J 8 *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_157886792!
dense_3/StatefulPartitionedCall
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_3/StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:Q M
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_4
·

&__inference_signature_wrapper_15789066
input_1
input_2
unknown
	unknown_0
identity˘StatefulPartitionedCallŬ
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
GPU2*0J 8 *,
f'R%
#__inference__wrapped_model_157886042
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
?

*__inference_model_3_layer_call_fn_15789416
inputs_0
inputs_1
unknown
	unknown_0
identity˘StatefulPartitionedCall
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
GPU2*0J 8 *N
fIRG
E__inference_model_3_layer_call_and_return_conditional_losses_157889542
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
î,
£
__inference_call_15789344
input_0
input_1?
;model_3_sequential_3_dense_3_matmul_readvariableop_resource@
<model_3_sequential_3_dense_3_biasadd_readvariableop_resource
identity˘3model_3/sequential_3/dense_3/BiasAdd/ReadVariableOp˘5model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOp˘2model_3/sequential_3/dense_3/MatMul/ReadVariableOp˘4model_3/sequential_3/dense_3/MatMul_1/ReadVariableOp
'model_3/sequential_3/dropout_3/IdentityIdentityinput_0*
T0* 
_output_shapes
:
2)
'model_3/sequential_3/dropout_3/Identity
=model_3/sequential_3/one_hot_encoding_layer_3/PartitionedCallPartitionedCall0model_3/sequential_3/dropout_3/Identity:output:0*
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
GPU2*0J 8 *"
fR
__inference_call_157892672?
=model_3/sequential_3/one_hot_encoding_layer_3/PartitionedCall
$model_3/sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2&
$model_3/sequential_3/flatten_3/Constŭ
&model_3/sequential_3/flatten_3/ReshapeReshapeFmodel_3/sequential_3/one_hot_encoding_layer_3/PartitionedCall:output:0-model_3/sequential_3/flatten_3/Const:output:0*
T0* 
_output_shapes
:
ô82(
&model_3/sequential_3/flatten_3/Reshapeĉ
2model_3/sequential_3/dense_3/MatMul/ReadVariableOpReadVariableOp;model_3_sequential_3_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype024
2model_3/sequential_3/dense_3/MatMul/ReadVariableOpì
#model_3/sequential_3/dense_3/MatMulMatMul/model_3/sequential_3/flatten_3/Reshape:output:0:model_3/sequential_3/dense_3/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2%
#model_3/sequential_3/dense_3/MatMulä
3model_3/sequential_3/dense_3/BiasAdd/ReadVariableOpReadVariableOp<model_3_sequential_3_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype025
3model_3/sequential_3/dense_3/BiasAdd/ReadVariableOpî
$model_3/sequential_3/dense_3/BiasAddBiasAdd-model_3/sequential_3/dense_3/MatMul:product:0;model_3/sequential_3/dense_3/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2&
$model_3/sequential_3/dense_3/BiasAdd
)model_3/sequential_3/dropout_3/Identity_1Identityinput_1*
T0* 
_output_shapes
:
2+
)model_3/sequential_3/dropout_3/Identity_1£
?model_3/sequential_3/one_hot_encoding_layer_3/PartitionedCall_1PartitionedCall2model_3/sequential_3/dropout_3/Identity_1:output:0*
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
GPU2*0J 8 *"
fR
__inference_call_157892672A
?model_3/sequential_3/one_hot_encoding_layer_3/PartitionedCall_1Ħ
&model_3/sequential_3/flatten_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2(
&model_3/sequential_3/flatten_3/Const_1
(model_3/sequential_3/flatten_3/Reshape_1ReshapeHmodel_3/sequential_3/one_hot_encoding_layer_3/PartitionedCall_1:output:0/model_3/sequential_3/flatten_3/Const_1:output:0*
T0* 
_output_shapes
:
ô82*
(model_3/sequential_3/flatten_3/Reshape_1ê
4model_3/sequential_3/dense_3/MatMul_1/ReadVariableOpReadVariableOp;model_3_sequential_3_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype026
4model_3/sequential_3/dense_3/MatMul_1/ReadVariableOpô
%model_3/sequential_3/dense_3/MatMul_1MatMul1model_3/sequential_3/flatten_3/Reshape_1:output:0<model_3/sequential_3/dense_3/MatMul_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2'
%model_3/sequential_3/dense_3/MatMul_1è
5model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOpReadVariableOp<model_3_sequential_3_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype027
5model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOpö
&model_3/sequential_3/dense_3/BiasAdd_1BiasAdd/model_3/sequential_3/dense_3/MatMul_1:product:0=model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2(
&model_3/sequential_3/dense_3/BiasAdd_1
(model_3/distance_layer_3/PartitionedCallPartitionedCall-model_3/sequential_3/dense_3/BiasAdd:output:0/model_3/sequential_3/dense_3/BiasAdd_1:output:0*
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
GPU2*0J 8 *"
fR
__inference_call_157893412*
(model_3/distance_layer_3/PartitionedCallÓ
IdentityIdentity1model_3/distance_layer_3/PartitionedCall:output:04^model_3/sequential_3/dense_3/BiasAdd/ReadVariableOp6^model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOp3^model_3/sequential_3/dense_3/MatMul/ReadVariableOp5^model_3/sequential_3/dense_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes	
:2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :
:
::2j
3model_3/sequential_3/dense_3/BiasAdd/ReadVariableOp3model_3/sequential_3/dense_3/BiasAdd/ReadVariableOp2n
5model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOp5model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOp2h
2model_3/sequential_3/dense_3/MatMul/ReadVariableOp2model_3/sequential_3/dense_3/MatMul/ReadVariableOp2l
4model_3/sequential_3/dense_3/MatMul_1/ReadVariableOp4model_3/sequential_3/dense_3/MatMul_1/ReadVariableOp:I E
 
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
í

2__inference_siamese_model_3_layer_call_fn_15789220
input_0
input_1
unknown
	unknown_0
identity˘StatefulPartitionedCall
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
GPU2*0J 8 *V
fQRO
M__inference_siamese_model_3_layer_call_and_return_conditional_losses_157890292
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


J__inference_sequential_3_layer_call_and_return_conditional_losses_15789529

inputs*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identity˘dense_3/BiasAdd/ReadVariableOp˘dense_3/MatMul/ReadVariableOpw
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout_3/dropout/Const
dropout_3/dropout/MulMulinputs dropout_3/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout_3/dropout/Mulh
dropout_3/dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout_3/dropout/ShapeÓ
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype020
.dropout_3/dropout/random_uniform/RandomUniform
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dropout_3/dropout/GreaterEqual/yç
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
dropout_3/dropout/GreaterEqual
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout_3/dropout/Cast£
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout_3/dropout/Mul_1ĉ
(one_hot_encoding_layer_3/PartitionedCallPartitionedCalldropout_3/dropout/Mul_1:z:0*
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
GPU2*0J 8 *"
fR
__inference_call_157885202*
(one_hot_encoding_layer_3/PartitionedCalls
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2
flatten_3/Constħ
flatten_3/ReshapeReshape1one_hot_encoding_layer_3/PartitionedCall:output:0flatten_3/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82
flatten_3/Reshape§
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02
dense_3/MatMul/ReadVariableOp 
dense_3/MatMulMatMulflatten_3/Reshape:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp˘
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_3/BiasAdd?
IdentityIdentitydense_3/BiasAdd:output:0^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
½
m
V__inference_one_hot_encoding_layer_3_layer_call_and_return_conditional_losses_15788647
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
Ĝ)
Ħ
E__inference_model_3_layer_call_and_return_conditional_losses_15789488
inputs_0
inputs_17
3sequential_3_dense_3_matmul_readvariableop_resource8
4sequential_3_dense_3_biasadd_readvariableop_resource
identity˘+sequential_3/dense_3/BiasAdd/ReadVariableOp˘-sequential_3/dense_3/BiasAdd_1/ReadVariableOp˘*sequential_3/dense_3/MatMul/ReadVariableOp˘,sequential_3/dense_3/MatMul_1/ReadVariableOp
sequential_3/dropout_3/IdentityIdentityinputs_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2!
sequential_3/dropout_3/Identity
5sequential_3/one_hot_encoding_layer_3/PartitionedCallPartitionedCall(sequential_3/dropout_3/Identity:output:0*
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
GPU2*0J 8 *"
fR
__inference_call_1578852027
5sequential_3/one_hot_encoding_layer_3/PartitionedCall
sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2
sequential_3/flatten_3/Constċ
sequential_3/flatten_3/ReshapeReshape>sequential_3/one_hot_encoding_layer_3/PartitionedCall:output:0%sequential_3/flatten_3/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82 
sequential_3/flatten_3/ReshapeÎ
*sequential_3/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02,
*sequential_3/dense_3/MatMul/ReadVariableOpÔ
sequential_3/dense_3/MatMulMatMul'sequential_3/flatten_3/Reshape:output:02sequential_3/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_3/dense_3/MatMulÌ
+sequential_3/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+sequential_3/dense_3/BiasAdd/ReadVariableOpÖ
sequential_3/dense_3/BiasAddBiasAdd%sequential_3/dense_3/MatMul:product:03sequential_3/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_3/dense_3/BiasAdd
!sequential_3/dropout_3/Identity_1Identityinputs_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2#
!sequential_3/dropout_3/Identity_1
7sequential_3/one_hot_encoding_layer_3/PartitionedCall_1PartitionedCall*sequential_3/dropout_3/Identity_1:output:0*
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
GPU2*0J 8 *"
fR
__inference_call_1578852029
7sequential_3/one_hot_encoding_layer_3/PartitionedCall_1
sequential_3/flatten_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2 
sequential_3/flatten_3/Const_1í
 sequential_3/flatten_3/Reshape_1Reshape@sequential_3/one_hot_encoding_layer_3/PartitionedCall_1:output:0'sequential_3/flatten_3/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82"
 sequential_3/flatten_3/Reshape_1Ò
,sequential_3/dense_3/MatMul_1/ReadVariableOpReadVariableOp3sequential_3_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02.
,sequential_3/dense_3/MatMul_1/ReadVariableOpÜ
sequential_3/dense_3/MatMul_1MatMul)sequential_3/flatten_3/Reshape_1:output:04sequential_3/dense_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_3/dense_3/MatMul_1?
-sequential_3/dense_3/BiasAdd_1/ReadVariableOpReadVariableOp4sequential_3_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-sequential_3/dense_3/BiasAdd_1/ReadVariableOpŜ
sequential_3/dense_3/BiasAdd_1BiasAdd'sequential_3/dense_3/MatMul_1:product:05sequential_3/dense_3/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
sequential_3/dense_3/BiasAdd_1
 distance_layer_3/PartitionedCallPartitionedCall%sequential_3/dense_3/BiasAdd:output:0'sequential_3/dense_3/BiasAdd_1:output:0*
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
GPU2*0J 8 *"
fR
__inference_call_157885942"
 distance_layer_3/PartitionedCall³
IdentityIdentity)distance_layer_3/PartitionedCall:output:0,^sequential_3/dense_3/BiasAdd/ReadVariableOp.^sequential_3/dense_3/BiasAdd_1/ReadVariableOp+^sequential_3/dense_3/MatMul/ReadVariableOp-^sequential_3/dense_3/MatMul_1/ReadVariableOp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2Z
+sequential_3/dense_3/BiasAdd/ReadVariableOp+sequential_3/dense_3/BiasAdd/ReadVariableOp2^
-sequential_3/dense_3/BiasAdd_1/ReadVariableOp-sequential_3/dense_3/BiasAdd_1/ReadVariableOp2X
*sequential_3/dense_3/MatMul/ReadVariableOp*sequential_3/dense_3/MatMul/ReadVariableOp2\
,sequential_3/dense_3/MatMul_1/ReadVariableOp,sequential_3/dense_3/MatMul_1/ReadVariableOp:R N
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
ż 
9
__inference_call_15789681
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

LogicalNotİ
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
 *&
else_branchR
cond_false_15789664*
output_shapes	
:*%
then_branchR
cond_true_157896632
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
ĝ

J__inference_sequential_3_layer_call_and_return_conditional_losses_15789543

inputs*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identity˘dense_3/BiasAdd/ReadVariableOp˘dense_3/MatMul/ReadVariableOpo
dropout_3/IdentityIdentityinputs*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout_3/Identityĉ
(one_hot_encoding_layer_3/PartitionedCallPartitionedCalldropout_3/Identity:output:0*
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
GPU2*0J 8 *"
fR
__inference_call_157885202*
(one_hot_encoding_layer_3/PartitionedCalls
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2
flatten_3/Constħ
flatten_3/ReshapeReshape1one_hot_encoding_layer_3/PartitionedCall:output:0flatten_3/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82
flatten_3/Reshape§
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02
dense_3/MatMul/ReadVariableOp 
dense_3/MatMulMatMulflatten_3/Reshape:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp˘
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_3/BiasAdd?
IdentityIdentitydense_3/BiasAdd:output:0^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

?
cond_false_15789664
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
Ô"
n
N__inference_distance_layer_3_layer_call_and_return_conditional_losses_15788839
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

LogicalNotı
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
 *&
else_branchR
cond_false_15788822*"
output_shapes
:˙˙˙˙˙˙˙˙˙*%
then_branchR
cond_true_157888212
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
ó

/__inference_sequential_3_layer_call_fn_15788751
input_4
unknown
	unknown_0
identity˘StatefulPartitionedCall˙
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0*
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
GPU2*0J 8 *S
fNRL
J__inference_sequential_3_layer_call_and_return_conditional_losses_157887442
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
_user_specified_name	input_4
¨
ĝ
E__inference_model_3_layer_call_and_return_conditional_losses_15788905

inputs
inputs_1
sequential_3_15788895
sequential_3_15788897
identity˘$sequential_3/StatefulPartitionedCall˘&sequential_3/StatefulPartitionedCall_1²
$sequential_3/StatefulPartitionedCallStatefulPartitionedCallinputssequential_3_15788895sequential_3_15788897*
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
GPU2*0J 8 *S
fNRL
J__inference_sequential_3_layer_call_and_return_conditional_losses_157887442&
$sequential_3/StatefulPartitionedCall¸
&sequential_3/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1sequential_3_15788895sequential_3_15788897*
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
GPU2*0J 8 *S
fNRL
J__inference_sequential_3_layer_call_and_return_conditional_losses_157887442(
&sequential_3/StatefulPartitionedCall_1Ĉ
 distance_layer_3/PartitionedCallPartitionedCall-sequential_3/StatefulPartitionedCall:output:0/sequential_3/StatefulPartitionedCall_1:output:0*
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
GPU2*0J 8 *W
fRRP
N__inference_distance_layer_3_layer_call_and_return_conditional_losses_157888392"
 distance_layer_3/PartitionedCallÉ
IdentityIdentity)distance_layer_3/PartitionedCall:output:0%^sequential_3/StatefulPartitionedCall'^sequential_3/StatefulPartitionedCall_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall2P
&sequential_3/StatefulPartitionedCall_1&sequential_3/StatefulPartitionedCall_1:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:PL
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ĥ
?
cond_false_15789601
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

9
cond_true_15788821
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
Ĥ
?
cond_false_15788577
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
?

/__inference_sequential_3_layer_call_fn_15789552

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallŝ
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
GPU2*0J 8 *S
fNRL
J__inference_sequential_3_layer_call_and_return_conditional_losses_157887232
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
Ŝ@
Ħ
E__inference_model_3_layer_call_and_return_conditional_losses_15789382
inputs_0
inputs_17
3sequential_3_dense_3_matmul_readvariableop_resource8
4sequential_3_dense_3_biasadd_readvariableop_resource
identity˘+sequential_3/dense_3/BiasAdd/ReadVariableOp˘-sequential_3/dense_3/BiasAdd_1/ReadVariableOp˘*sequential_3/dense_3/MatMul/ReadVariableOp˘,sequential_3/dense_3/MatMul_1/ReadVariableOp
$sequential_3/dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2&
$sequential_3/dropout_3/dropout/Constğ
"sequential_3/dropout_3/dropout/MulMulinputs_0-sequential_3/dropout_3/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2$
"sequential_3/dropout_3/dropout/Mul
$sequential_3/dropout_3/dropout/ShapeShapeinputs_0*
T0*
_output_shapes
:2&
$sequential_3/dropout_3/dropout/Shapeú
;sequential_3/dropout_3/dropout/random_uniform/RandomUniformRandomUniform-sequential_3/dropout_3/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02=
;sequential_3/dropout_3/dropout/random_uniform/RandomUniform£
-sequential_3/dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2/
-sequential_3/dropout_3/dropout/GreaterEqual/y
+sequential_3/dropout_3/dropout/GreaterEqualGreaterEqualDsequential_3/dropout_3/dropout/random_uniform/RandomUniform:output:06sequential_3/dropout_3/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2-
+sequential_3/dropout_3/dropout/GreaterEqualĊ
#sequential_3/dropout_3/dropout/CastCast/sequential_3/dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#sequential_3/dropout_3/dropout/Cast×
$sequential_3/dropout_3/dropout/Mul_1Mul&sequential_3/dropout_3/dropout/Mul:z:0'sequential_3/dropout_3/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2&
$sequential_3/dropout_3/dropout/Mul_1
5sequential_3/one_hot_encoding_layer_3/PartitionedCallPartitionedCall(sequential_3/dropout_3/dropout/Mul_1:z:0*
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
GPU2*0J 8 *"
fR
__inference_call_1578852027
5sequential_3/one_hot_encoding_layer_3/PartitionedCall
sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2
sequential_3/flatten_3/Constċ
sequential_3/flatten_3/ReshapeReshape>sequential_3/one_hot_encoding_layer_3/PartitionedCall:output:0%sequential_3/flatten_3/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82 
sequential_3/flatten_3/ReshapeÎ
*sequential_3/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02,
*sequential_3/dense_3/MatMul/ReadVariableOpÔ
sequential_3/dense_3/MatMulMatMul'sequential_3/flatten_3/Reshape:output:02sequential_3/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_3/dense_3/MatMulÌ
+sequential_3/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+sequential_3/dense_3/BiasAdd/ReadVariableOpÖ
sequential_3/dense_3/BiasAddBiasAdd%sequential_3/dense_3/MatMul:product:03sequential_3/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_3/dense_3/BiasAdd
&sequential_3/dropout_3/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2(
&sequential_3/dropout_3/dropout_1/ConstÁ
$sequential_3/dropout_3/dropout_1/MulMulinputs_1/sequential_3/dropout_3/dropout_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2&
$sequential_3/dropout_3/dropout_1/Mul
&sequential_3/dropout_3/dropout_1/ShapeShapeinputs_1*
T0*
_output_shapes
:2(
&sequential_3/dropout_3/dropout_1/Shape
=sequential_3/dropout_3/dropout_1/random_uniform/RandomUniformRandomUniform/sequential_3/dropout_3/dropout_1/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02?
=sequential_3/dropout_3/dropout_1/random_uniform/RandomUniform§
/sequential_3/dropout_3/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    21
/sequential_3/dropout_3/dropout_1/GreaterEqual/y£
-sequential_3/dropout_3/dropout_1/GreaterEqualGreaterEqualFsequential_3/dropout_3/dropout_1/random_uniform/RandomUniform:output:08sequential_3/dropout_3/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2/
-sequential_3/dropout_3/dropout_1/GreaterEqualË
%sequential_3/dropout_3/dropout_1/CastCast1sequential_3/dropout_3/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%sequential_3/dropout_3/dropout_1/Castß
&sequential_3/dropout_3/dropout_1/Mul_1Mul(sequential_3/dropout_3/dropout_1/Mul:z:0)sequential_3/dropout_3/dropout_1/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2(
&sequential_3/dropout_3/dropout_1/Mul_1
7sequential_3/one_hot_encoding_layer_3/PartitionedCall_1PartitionedCall*sequential_3/dropout_3/dropout_1/Mul_1:z:0*
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
GPU2*0J 8 *"
fR
__inference_call_1578852029
7sequential_3/one_hot_encoding_layer_3/PartitionedCall_1
sequential_3/flatten_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2 
sequential_3/flatten_3/Const_1í
 sequential_3/flatten_3/Reshape_1Reshape@sequential_3/one_hot_encoding_layer_3/PartitionedCall_1:output:0'sequential_3/flatten_3/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82"
 sequential_3/flatten_3/Reshape_1Ò
,sequential_3/dense_3/MatMul_1/ReadVariableOpReadVariableOp3sequential_3_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02.
,sequential_3/dense_3/MatMul_1/ReadVariableOpÜ
sequential_3/dense_3/MatMul_1MatMul)sequential_3/flatten_3/Reshape_1:output:04sequential_3/dense_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_3/dense_3/MatMul_1?
-sequential_3/dense_3/BiasAdd_1/ReadVariableOpReadVariableOp4sequential_3_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-sequential_3/dense_3/BiasAdd_1/ReadVariableOpŜ
sequential_3/dense_3/BiasAdd_1BiasAdd'sequential_3/dense_3/MatMul_1:product:05sequential_3/dense_3/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
sequential_3/dense_3/BiasAdd_1
 distance_layer_3/PartitionedCallPartitionedCall%sequential_3/dense_3/BiasAdd:output:0'sequential_3/dense_3/BiasAdd_1:output:0*
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
GPU2*0J 8 *"
fR
__inference_call_157885942"
 distance_layer_3/PartitionedCall³
IdentityIdentity)distance_layer_3/PartitionedCall:output:0,^sequential_3/dense_3/BiasAdd/ReadVariableOp.^sequential_3/dense_3/BiasAdd_1/ReadVariableOp+^sequential_3/dense_3/MatMul/ReadVariableOp-^sequential_3/dense_3/MatMul_1/ReadVariableOp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2Z
+sequential_3/dense_3/BiasAdd/ReadVariableOp+sequential_3/dense_3/BiasAdd/ReadVariableOp2^
-sequential_3/dense_3/BiasAdd_1/ReadVariableOp-sequential_3/dense_3/BiasAdd_1/ReadVariableOp2X
*sequential_3/dense_3/MatMul/ReadVariableOp*sequential_3/dense_3/MatMul/ReadVariableOp2\
,sequential_3/dense_3/MatMul_1/ReadVariableOp,sequential_3/dense_3/MatMul_1/ReadVariableOp:R N
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
ß%
Ê
!__inference__traced_save_15789890
file_prefix$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop4
0savev2_adam_dense_3_kernel_m_read_readvariableop2
.savev2_adam_dense_3_bias_m_read_readvariableop4
0savev2_adam_dense_3_kernel_v_read_readvariableop2
.savev2_adam_dense_3_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0 savev2_total_read_readvariableop savev2_count_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop0savev2_adam_dense_3_kernel_m_read_readvariableop.savev2_adam_dense_3_bias_m_read_readvariableop0savev2_adam_dense_3_kernel_v_read_readvariableop.savev2_adam_dense_3_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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

S
3__inference_distance_layer_3_layer_call_fn_15789624
s1
s2
identityÌ
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
GPU2*0J 8 *W
fRRP
N__inference_distance_layer_3_layer_call_and_return_conditional_losses_157888392
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
â
9
cond_true_15789323
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
ó

/__inference_sequential_3_layer_call_fn_15788730
input_4
unknown
	unknown_0
identity˘StatefulPartitionedCall˙
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0*
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
GPU2*0J 8 *S
fNRL
J__inference_sequential_3_layer_call_and_return_conditional_losses_157887232
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
_user_specified_name	input_4
Â
Û
J__inference_sequential_3_layer_call_and_return_conditional_losses_15788723

inputs
dense_3_15788717
dense_3_15788719
identity˘dense_3/StatefulPartitionedCall˘!dropout_3/StatefulPartitionedCallġ
!dropout_3/StatefulPartitionedCallStatefulPartitionedCallinputs*
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
GPU2*0J 8 *P
fKRI
G__inference_dropout_3_layer_call_and_return_conditional_losses_157886202#
!dropout_3/StatefulPartitionedCall²
(one_hot_encoding_layer_3/PartitionedCallPartitionedCall*dropout_3/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *_
fZRX
V__inference_one_hot_encoding_layer_3_layer_call_and_return_conditional_losses_157886472*
(one_hot_encoding_layer_3/PartitionedCall
flatten_3/PartitionedCallPartitionedCall1one_hot_encoding_layer_3/PartitionedCall:output:0*
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
GPU2*0J 8 *P
fKRI
G__inference_flatten_3_layer_call_and_return_conditional_losses_157886612
flatten_3/PartitionedCallµ
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_3_15788717dense_3_15788719*
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
GPU2*0J 8 *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_157886792!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_3/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
¤9
Ò
$__inference__traced_restore_15789939
file_prefix
assignvariableop_total
assignvariableop_1_count 
assignvariableop_2_adam_iter"
assignvariableop_3_adam_beta_1"
assignvariableop_4_adam_beta_2!
assignvariableop_5_adam_decay)
%assignvariableop_6_adam_learning_rate%
!assignvariableop_7_dense_3_kernel#
assignvariableop_8_dense_3_bias,
(assignvariableop_9_adam_dense_3_kernel_m+
'assignvariableop_10_adam_dense_3_bias_m-
)assignvariableop_11_adam_dense_3_kernel_v+
'assignvariableop_12_adam_dense_3_bias_v
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
AssignVariableOp_7AssignVariableOp!assignvariableop_7_dense_3_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¤
AssignVariableOp_8AssignVariableOpassignvariableop_8_dense_3_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9­
AssignVariableOp_9AssignVariableOp(assignvariableop_9_adam_dense_3_kernel_mIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ż
AssignVariableOp_10AssignVariableOp'assignvariableop_10_adam_dense_3_bias_mIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11ħ
AssignVariableOp_11AssignVariableOp)assignvariableop_11_adam_dense_3_kernel_vIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Ż
AssignVariableOp_12AssignVariableOp'assignvariableop_12_adam_dense_3_bias_vIdentity_12:output:0"/device:CPU:0*
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
öG
×
M__inference_siamese_model_3_layer_call_and_return_conditional_losses_15789104
input_1
input_2?
;model_3_sequential_3_dense_3_matmul_readvariableop_resource@
<model_3_sequential_3_dense_3_biasadd_readvariableop_resource
identity˘3model_3/sequential_3/dense_3/BiasAdd/ReadVariableOp˘5model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOp˘2model_3/sequential_3/dense_3/MatMul/ReadVariableOp˘4model_3/sequential_3/dense_3/MatMul_1/ReadVariableOpĦ
,model_3/sequential_3/dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2.
,model_3/sequential_3/dropout_3/dropout/ConstÒ
*model_3/sequential_3/dropout_3/dropout/MulMulinput_15model_3/sequential_3/dropout_3/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2,
*model_3/sequential_3/dropout_3/dropout/Mul
,model_3/sequential_3/dropout_3/dropout/ShapeShapeinput_1*
T0*
_output_shapes
:2.
,model_3/sequential_3/dropout_3/dropout/Shape
Cmodel_3/sequential_3/dropout_3/dropout/random_uniform/RandomUniformRandomUniform5model_3/sequential_3/dropout_3/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02E
Cmodel_3/sequential_3/dropout_3/dropout/random_uniform/RandomUniform³
5model_3/sequential_3/dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5model_3/sequential_3/dropout_3/dropout/GreaterEqual/yğ
3model_3/sequential_3/dropout_3/dropout/GreaterEqualGreaterEqualLmodel_3/sequential_3/dropout_3/dropout/random_uniform/RandomUniform:output:0>model_3/sequential_3/dropout_3/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙25
3model_3/sequential_3/dropout_3/dropout/GreaterEqualŬ
+model_3/sequential_3/dropout_3/dropout/CastCast7model_3/sequential_3/dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2-
+model_3/sequential_3/dropout_3/dropout/Cast÷
,model_3/sequential_3/dropout_3/dropout/Mul_1Mul.model_3/sequential_3/dropout_3/dropout/Mul:z:0/model_3/sequential_3/dropout_3/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2.
,model_3/sequential_3/dropout_3/dropout/Mul_1?
=model_3/sequential_3/one_hot_encoding_layer_3/PartitionedCallPartitionedCall0model_3/sequential_3/dropout_3/dropout/Mul_1:z:0*
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
GPU2*0J 8 *"
fR
__inference_call_157885202?
=model_3/sequential_3/one_hot_encoding_layer_3/PartitionedCall
$model_3/sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2&
$model_3/sequential_3/flatten_3/Const
&model_3/sequential_3/flatten_3/ReshapeReshapeFmodel_3/sequential_3/one_hot_encoding_layer_3/PartitionedCall:output:0-model_3/sequential_3/flatten_3/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82(
&model_3/sequential_3/flatten_3/Reshapeĉ
2model_3/sequential_3/dense_3/MatMul/ReadVariableOpReadVariableOp;model_3_sequential_3_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype024
2model_3/sequential_3/dense_3/MatMul/ReadVariableOpô
#model_3/sequential_3/dense_3/MatMulMatMul/model_3/sequential_3/flatten_3/Reshape:output:0:model_3/sequential_3/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#model_3/sequential_3/dense_3/MatMulä
3model_3/sequential_3/dense_3/BiasAdd/ReadVariableOpReadVariableOp<model_3_sequential_3_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype025
3model_3/sequential_3/dense_3/BiasAdd/ReadVariableOpö
$model_3/sequential_3/dense_3/BiasAddBiasAdd-model_3/sequential_3/dense_3/MatMul:product:0;model_3/sequential_3/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2&
$model_3/sequential_3/dense_3/BiasAdd?
.model_3/sequential_3/dropout_3/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?20
.model_3/sequential_3/dropout_3/dropout_1/ConstĜ
,model_3/sequential_3/dropout_3/dropout_1/MulMulinput_27model_3/sequential_3/dropout_3/dropout_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2.
,model_3/sequential_3/dropout_3/dropout_1/Mul
.model_3/sequential_3/dropout_3/dropout_1/ShapeShapeinput_2*
T0*
_output_shapes
:20
.model_3/sequential_3/dropout_3/dropout_1/Shape
Emodel_3/sequential_3/dropout_3/dropout_1/random_uniform/RandomUniformRandomUniform7model_3/sequential_3/dropout_3/dropout_1/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02G
Emodel_3/sequential_3/dropout_3/dropout_1/random_uniform/RandomUniform·
7model_3/sequential_3/dropout_3/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    29
7model_3/sequential_3/dropout_3/dropout_1/GreaterEqual/y?
5model_3/sequential_3/dropout_3/dropout_1/GreaterEqualGreaterEqualNmodel_3/sequential_3/dropout_3/dropout_1/random_uniform/RandomUniform:output:0@model_3/sequential_3/dropout_3/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙27
5model_3/sequential_3/dropout_3/dropout_1/GreaterEqual?
-model_3/sequential_3/dropout_3/dropout_1/CastCast9model_3/sequential_3/dropout_3/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2/
-model_3/sequential_3/dropout_3/dropout_1/Cast˙
.model_3/sequential_3/dropout_3/dropout_1/Mul_1Mul0model_3/sequential_3/dropout_3/dropout_1/Mul:z:01model_3/sequential_3/dropout_3/dropout_1/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙20
.model_3/sequential_3/dropout_3/dropout_1/Mul_1Ğ
?model_3/sequential_3/one_hot_encoding_layer_3/PartitionedCall_1PartitionedCall2model_3/sequential_3/dropout_3/dropout_1/Mul_1:z:0*
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
GPU2*0J 8 *"
fR
__inference_call_157885202A
?model_3/sequential_3/one_hot_encoding_layer_3/PartitionedCall_1Ħ
&model_3/sequential_3/flatten_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2(
&model_3/sequential_3/flatten_3/Const_1
(model_3/sequential_3/flatten_3/Reshape_1ReshapeHmodel_3/sequential_3/one_hot_encoding_layer_3/PartitionedCall_1:output:0/model_3/sequential_3/flatten_3/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82*
(model_3/sequential_3/flatten_3/Reshape_1ê
4model_3/sequential_3/dense_3/MatMul_1/ReadVariableOpReadVariableOp;model_3_sequential_3_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype026
4model_3/sequential_3/dense_3/MatMul_1/ReadVariableOpü
%model_3/sequential_3/dense_3/MatMul_1MatMul1model_3/sequential_3/flatten_3/Reshape_1:output:0<model_3/sequential_3/dense_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%model_3/sequential_3/dense_3/MatMul_1è
5model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOpReadVariableOp<model_3_sequential_3_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype027
5model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOpŝ
&model_3/sequential_3/dense_3/BiasAdd_1BiasAdd/model_3/sequential_3/dense_3/MatMul_1:product:0=model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2(
&model_3/sequential_3/dense_3/BiasAdd_1Ħ
(model_3/distance_layer_3/PartitionedCallPartitionedCall-model_3/sequential_3/dense_3/BiasAdd:output:0/model_3/sequential_3/dense_3/BiasAdd_1:output:0*
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
GPU2*0J 8 *"
fR
__inference_call_157885942*
(model_3/distance_layer_3/PartitionedCallÛ
IdentityIdentity1model_3/distance_layer_3/PartitionedCall:output:04^model_3/sequential_3/dense_3/BiasAdd/ReadVariableOp6^model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOp3^model_3/sequential_3/dense_3/MatMul/ReadVariableOp5^model_3/sequential_3/dense_3/MatMul_1/ReadVariableOp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2j
3model_3/sequential_3/dense_3/BiasAdd/ReadVariableOp3model_3/sequential_3/dense_3/BiasAdd/ReadVariableOp2n
5model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOp5model_3/sequential_3/dense_3/BiasAdd_1/ReadVariableOp2h
2model_3/sequential_3/dense_3/MatMul/ReadVariableOp2model_3/sequential_3/dense_3/MatMul/ReadVariableOp2l
4model_3/sequential_3/dense_3/MatMul_1/ReadVariableOp4model_3/sequential_3/dense_3/MatMul_1/ReadVariableOp:Q M
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
Ŝ@
Ħ
E__inference_model_3_layer_call_and_return_conditional_losses_15789464
inputs_0
inputs_17
3sequential_3_dense_3_matmul_readvariableop_resource8
4sequential_3_dense_3_biasadd_readvariableop_resource
identity˘+sequential_3/dense_3/BiasAdd/ReadVariableOp˘-sequential_3/dense_3/BiasAdd_1/ReadVariableOp˘*sequential_3/dense_3/MatMul/ReadVariableOp˘,sequential_3/dense_3/MatMul_1/ReadVariableOp
$sequential_3/dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2&
$sequential_3/dropout_3/dropout/Constğ
"sequential_3/dropout_3/dropout/MulMulinputs_0-sequential_3/dropout_3/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2$
"sequential_3/dropout_3/dropout/Mul
$sequential_3/dropout_3/dropout/ShapeShapeinputs_0*
T0*
_output_shapes
:2&
$sequential_3/dropout_3/dropout/Shapeú
;sequential_3/dropout_3/dropout/random_uniform/RandomUniformRandomUniform-sequential_3/dropout_3/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02=
;sequential_3/dropout_3/dropout/random_uniform/RandomUniform£
-sequential_3/dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2/
-sequential_3/dropout_3/dropout/GreaterEqual/y
+sequential_3/dropout_3/dropout/GreaterEqualGreaterEqualDsequential_3/dropout_3/dropout/random_uniform/RandomUniform:output:06sequential_3/dropout_3/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2-
+sequential_3/dropout_3/dropout/GreaterEqualĊ
#sequential_3/dropout_3/dropout/CastCast/sequential_3/dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#sequential_3/dropout_3/dropout/Cast×
$sequential_3/dropout_3/dropout/Mul_1Mul&sequential_3/dropout_3/dropout/Mul:z:0'sequential_3/dropout_3/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2&
$sequential_3/dropout_3/dropout/Mul_1
5sequential_3/one_hot_encoding_layer_3/PartitionedCallPartitionedCall(sequential_3/dropout_3/dropout/Mul_1:z:0*
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
GPU2*0J 8 *"
fR
__inference_call_1578852027
5sequential_3/one_hot_encoding_layer_3/PartitionedCall
sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2
sequential_3/flatten_3/Constċ
sequential_3/flatten_3/ReshapeReshape>sequential_3/one_hot_encoding_layer_3/PartitionedCall:output:0%sequential_3/flatten_3/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82 
sequential_3/flatten_3/ReshapeÎ
*sequential_3/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02,
*sequential_3/dense_3/MatMul/ReadVariableOpÔ
sequential_3/dense_3/MatMulMatMul'sequential_3/flatten_3/Reshape:output:02sequential_3/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_3/dense_3/MatMulÌ
+sequential_3/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+sequential_3/dense_3/BiasAdd/ReadVariableOpÖ
sequential_3/dense_3/BiasAddBiasAdd%sequential_3/dense_3/MatMul:product:03sequential_3/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_3/dense_3/BiasAdd
&sequential_3/dropout_3/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2(
&sequential_3/dropout_3/dropout_1/ConstÁ
$sequential_3/dropout_3/dropout_1/MulMulinputs_1/sequential_3/dropout_3/dropout_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2&
$sequential_3/dropout_3/dropout_1/Mul
&sequential_3/dropout_3/dropout_1/ShapeShapeinputs_1*
T0*
_output_shapes
:2(
&sequential_3/dropout_3/dropout_1/Shape
=sequential_3/dropout_3/dropout_1/random_uniform/RandomUniformRandomUniform/sequential_3/dropout_3/dropout_1/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02?
=sequential_3/dropout_3/dropout_1/random_uniform/RandomUniform§
/sequential_3/dropout_3/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    21
/sequential_3/dropout_3/dropout_1/GreaterEqual/y£
-sequential_3/dropout_3/dropout_1/GreaterEqualGreaterEqualFsequential_3/dropout_3/dropout_1/random_uniform/RandomUniform:output:08sequential_3/dropout_3/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2/
-sequential_3/dropout_3/dropout_1/GreaterEqualË
%sequential_3/dropout_3/dropout_1/CastCast1sequential_3/dropout_3/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%sequential_3/dropout_3/dropout_1/Castß
&sequential_3/dropout_3/dropout_1/Mul_1Mul(sequential_3/dropout_3/dropout_1/Mul:z:0)sequential_3/dropout_3/dropout_1/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2(
&sequential_3/dropout_3/dropout_1/Mul_1
7sequential_3/one_hot_encoding_layer_3/PartitionedCall_1PartitionedCall*sequential_3/dropout_3/dropout_1/Mul_1:z:0*
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
GPU2*0J 8 *"
fR
__inference_call_1578852029
7sequential_3/one_hot_encoding_layer_3/PartitionedCall_1
sequential_3/flatten_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2 
sequential_3/flatten_3/Const_1í
 sequential_3/flatten_3/Reshape_1Reshape@sequential_3/one_hot_encoding_layer_3/PartitionedCall_1:output:0'sequential_3/flatten_3/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82"
 sequential_3/flatten_3/Reshape_1Ò
,sequential_3/dense_3/MatMul_1/ReadVariableOpReadVariableOp3sequential_3_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02.
,sequential_3/dense_3/MatMul_1/ReadVariableOpÜ
sequential_3/dense_3/MatMul_1MatMul)sequential_3/flatten_3/Reshape_1:output:04sequential_3/dense_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_3/dense_3/MatMul_1?
-sequential_3/dense_3/BiasAdd_1/ReadVariableOpReadVariableOp4sequential_3_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-sequential_3/dense_3/BiasAdd_1/ReadVariableOpŜ
sequential_3/dense_3/BiasAdd_1BiasAdd'sequential_3/dense_3/MatMul_1:product:05sequential_3/dense_3/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
sequential_3/dense_3/BiasAdd_1
 distance_layer_3/PartitionedCallPartitionedCall%sequential_3/dense_3/BiasAdd:output:0'sequential_3/dense_3/BiasAdd_1:output:0*
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
GPU2*0J 8 *"
fR
__inference_call_157885942"
 distance_layer_3/PartitionedCall³
IdentityIdentity)distance_layer_3/PartitionedCall:output:0,^sequential_3/dense_3/BiasAdd/ReadVariableOp.^sequential_3/dense_3/BiasAdd_1/ReadVariableOp+^sequential_3/dense_3/MatMul/ReadVariableOp-^sequential_3/dense_3/MatMul_1/ReadVariableOp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2Z
+sequential_3/dense_3/BiasAdd/ReadVariableOp+sequential_3/dense_3/BiasAdd/ReadVariableOp2^
-sequential_3/dense_3/BiasAdd_1/ReadVariableOp-sequential_3/dense_3/BiasAdd_1/ReadVariableOp2X
*sequential_3/dense_3/MatMul/ReadVariableOp*sequential_3/dense_3/MatMul/ReadVariableOp2\
,sequential_3/dense_3/MatMul_1/ReadVariableOp,sequential_3/dense_3/MatMul_1/ReadVariableOp:R N
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
Ĝ)
Ħ
E__inference_model_3_layer_call_and_return_conditional_losses_15789406
inputs_0
inputs_17
3sequential_3_dense_3_matmul_readvariableop_resource8
4sequential_3_dense_3_biasadd_readvariableop_resource
identity˘+sequential_3/dense_3/BiasAdd/ReadVariableOp˘-sequential_3/dense_3/BiasAdd_1/ReadVariableOp˘*sequential_3/dense_3/MatMul/ReadVariableOp˘,sequential_3/dense_3/MatMul_1/ReadVariableOp
sequential_3/dropout_3/IdentityIdentityinputs_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2!
sequential_3/dropout_3/Identity
5sequential_3/one_hot_encoding_layer_3/PartitionedCallPartitionedCall(sequential_3/dropout_3/Identity:output:0*
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
GPU2*0J 8 *"
fR
__inference_call_1578852027
5sequential_3/one_hot_encoding_layer_3/PartitionedCall
sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2
sequential_3/flatten_3/Constċ
sequential_3/flatten_3/ReshapeReshape>sequential_3/one_hot_encoding_layer_3/PartitionedCall:output:0%sequential_3/flatten_3/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82 
sequential_3/flatten_3/ReshapeÎ
*sequential_3/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02,
*sequential_3/dense_3/MatMul/ReadVariableOpÔ
sequential_3/dense_3/MatMulMatMul'sequential_3/flatten_3/Reshape:output:02sequential_3/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_3/dense_3/MatMulÌ
+sequential_3/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+sequential_3/dense_3/BiasAdd/ReadVariableOpÖ
sequential_3/dense_3/BiasAddBiasAdd%sequential_3/dense_3/MatMul:product:03sequential_3/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_3/dense_3/BiasAdd
!sequential_3/dropout_3/Identity_1Identityinputs_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2#
!sequential_3/dropout_3/Identity_1
7sequential_3/one_hot_encoding_layer_3/PartitionedCall_1PartitionedCall*sequential_3/dropout_3/Identity_1:output:0*
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
GPU2*0J 8 *"
fR
__inference_call_1578852029
7sequential_3/one_hot_encoding_layer_3/PartitionedCall_1
sequential_3/flatten_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2 
sequential_3/flatten_3/Const_1í
 sequential_3/flatten_3/Reshape_1Reshape@sequential_3/one_hot_encoding_layer_3/PartitionedCall_1:output:0'sequential_3/flatten_3/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82"
 sequential_3/flatten_3/Reshape_1Ò
,sequential_3/dense_3/MatMul_1/ReadVariableOpReadVariableOp3sequential_3_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02.
,sequential_3/dense_3/MatMul_1/ReadVariableOpÜ
sequential_3/dense_3/MatMul_1MatMul)sequential_3/flatten_3/Reshape_1:output:04sequential_3/dense_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_3/dense_3/MatMul_1?
-sequential_3/dense_3/BiasAdd_1/ReadVariableOpReadVariableOp4sequential_3_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-sequential_3/dense_3/BiasAdd_1/ReadVariableOpŜ
sequential_3/dense_3/BiasAdd_1BiasAdd'sequential_3/dense_3/MatMul_1:product:05sequential_3/dense_3/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
sequential_3/dense_3/BiasAdd_1
 distance_layer_3/PartitionedCallPartitionedCall%sequential_3/dense_3/BiasAdd:output:0'sequential_3/dense_3/BiasAdd_1:output:0*
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
GPU2*0J 8 *"
fR
__inference_call_157885942"
 distance_layer_3/PartitionedCall³
IdentityIdentity)distance_layer_3/PartitionedCall:output:0,^sequential_3/dense_3/BiasAdd/ReadVariableOp.^sequential_3/dense_3/BiasAdd_1/ReadVariableOp+^sequential_3/dense_3/MatMul/ReadVariableOp-^sequential_3/dense_3/MatMul_1/ReadVariableOp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2Z
+sequential_3/dense_3/BiasAdd/ReadVariableOp+sequential_3/dense_3/BiasAdd/ReadVariableOp2^
-sequential_3/dense_3/BiasAdd_1/ReadVariableOp-sequential_3/dense_3/BiasAdd_1/ReadVariableOp2X
*sequential_3/dense_3/MatMul/ReadVariableOp*sequential_3/dense_3/MatMul/ReadVariableOp2\
,sequential_3/dense_3/MatMul_1/ReadVariableOp,sequential_3/dense_3/MatMul_1/ReadVariableOp:R N
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
inputs/1"ħL
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
StatefulPartitionedCall:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict:äë
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
_tf_keras_modelĴ{"class_name": "SiameseModel", "name": "siamese_model_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "SiameseModel"}, "training_config": {"loss": null, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "clipnorm": 1, "learning_rate": 1, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
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
_tf_keras_networkñ{"class_name": "Functional", "name": "model_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1821]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "sequence1"}, "name": "sequence1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1821]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "sequence2"}, "name": "sequence2", "inbound_nodes": []}, {"class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1821]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}, {"class_name": "OneHotEncodingLayer", "config": {"layer was saved without config": true}}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 910, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "name": "sequential_3", "inbound_nodes": [[["sequence1", 0, 0, {}]], [["sequence2", 0, 0, {}]]]}, {"class_name": "DistanceLayer", "config": {"layer was saved without config": true}, "name": "distance_layer_3", "inbound_nodes": [[["sequential_3", 1, 0, {"s2": ["sequential_3", 2, 0]}]]]}], "input_layers": [["sequence1", 0, 0], ["sequence2", 0, 0]], "output_layers": {"class_name": "__tuple__", "items": [["distance_layer_3", 0, 0]]}}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1821]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1821]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1821]}, {"class_name": "TensorShape", "items": [null, 1821]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}}
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
_tf_keras_sequential{"class_name": "Sequential", "name": "sequential_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1821]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}, {"class_name": "OneHotEncodingLayer", "config": {"layer was saved without config": true}}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 910, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1821]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential"}}
ı
*trainable_variables
+regularization_losses
,	variables
-	keras_api
n__call__
*o&call_and_return_all_conditional_losses
pcall" 
_tf_keras_layer{"class_name": "DistanceLayer", "name": "distance_layer_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
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
ô82dense_3/kernel
:2dense_3/bias
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
_tf_keras_layerĵ{"class_name": "Dropout", "name": "dropout_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}
Ç
7trainable_variables
8regularization_losses
9	variables
:	keras_api
s__call__
*t&call_and_return_all_conditional_losses
ucall"?
_tf_keras_layer{"class_name": "OneHotEncodingLayer", "name": "one_hot_encoding_layer_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
ĉ
;trainable_variables
<regularization_losses
=	variables
>	keras_api
v__call__
*w&call_and_return_all_conditional_losses"×
_tf_keras_layer½{"class_name": "Flatten", "name": "flatten_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
÷

kernel
bias
?trainable_variables
@regularization_losses
A	variables
B	keras_api
x__call__
*y&call_and_return_all_conditional_losses"Ò
_tf_keras_layer¸{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 910, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 7284}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7284]}}
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
ô82Adam/dense_3/kernel/m
 :2Adam/dense_3/bias/m
':%
ô82Adam/dense_3/kernel/v
 :2Adam/dense_3/bias/v
2
2__inference_siamese_model_3_layer_call_fn_15789230
2__inference_siamese_model_3_layer_call_fn_15789148
2__inference_siamese_model_3_layer_call_fn_15789138
2__inference_siamese_model_3_layer_call_fn_15789220²
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
ô2ñ
M__inference_siamese_model_3_layer_call_and_return_conditional_losses_15789104
M__inference_siamese_model_3_layer_call_and_return_conditional_losses_15789210
M__inference_siamese_model_3_layer_call_and_return_conditional_losses_15789186
M__inference_siamese_model_3_layer_call_and_return_conditional_losses_15789128²
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
2
#__inference__wrapped_model_15788604à
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
Ŭ2Ú
__inference_call_15789254
__inference_call_15789344Ħ
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
Î2Ë
*__inference_model_3_layer_call_fn_15789508
*__inference_model_3_layer_call_fn_15789416
*__inference_model_3_layer_call_fn_15788888
*__inference_model_3_layer_call_fn_15789498
*__inference_model_3_layer_call_fn_15789426
*__inference_model_3_layer_call_fn_15788912À
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
?2í
E__inference_model_3_layer_call_and_return_conditional_losses_15789464
E__inference_model_3_layer_call_and_return_conditional_losses_15788849
E__inference_model_3_layer_call_and_return_conditional_losses_15789382
E__inference_model_3_layer_call_and_return_conditional_losses_15789406
E__inference_model_3_layer_call_and_return_conditional_losses_15789488
E__inference_model_3_layer_call_and_return_conditional_losses_15788863À
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
ÔBÑ
&__inference_signature_wrapper_15789066input_1input_2"
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
2
/__inference_sequential_3_layer_call_fn_15788730
/__inference_sequential_3_layer_call_fn_15788751
/__inference_sequential_3_layer_call_fn_15789552
/__inference_sequential_3_layer_call_fn_15789561À
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
ö2ó
J__inference_sequential_3_layer_call_and_return_conditional_losses_15788708
J__inference_sequential_3_layer_call_and_return_conditional_losses_15789543
J__inference_sequential_3_layer_call_and_return_conditional_losses_15788696
J__inference_sequential_3_layer_call_and_return_conditional_losses_15789529À
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
ß2Ü
3__inference_distance_layer_3_layer_call_fn_15789624¤
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
ú2÷
N__inference_distance_layer_3_layer_call_and_return_conditional_losses_15789618¤
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
à2Ŭ
__inference_call_15789681
__inference_call_15789738¤
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
2
,__inference_dropout_3_layer_call_fn_15789765
,__inference_dropout_3_layer_call_fn_15789760´
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
Ì2É
G__inference_dropout_3_layer_call_and_return_conditional_losses_15789750
G__inference_dropout_3_layer_call_and_return_conditional_losses_15789755´
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
à2Ŭ
;__inference_one_hot_encoding_layer_3_layer_call_fn_15789779
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
û2ĝ
V__inference_one_hot_encoding_layer_3_layer_call_and_return_conditional_losses_15789774
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
Ù2Ö
__inference_call_15789788
__inference_call_15789797
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
Ö2Ó
,__inference_flatten_3_layer_call_fn_15789808˘
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
ñ2î
G__inference_flatten_3_layer_call_and_return_conditional_losses_15789803˘
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
Ô2Ñ
*__inference_dense_3_layer_call_fn_15789827˘
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
ï2ì
E__inference_dense_3_layer_call_and_return_conditional_losses_15789818˘
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
 ı
#__inference__wrapped_model_15788604Z˘W
P˘M
K˘H
"
input_1˙˙˙˙˙˙˙˙˙
"
input_2˙˙˙˙˙˙˙˙˙
Ş "/Ş,
*
output_1
output_1˙˙˙˙˙˙˙˙˙
__inference_call_15789254~Z˘W
P˘M
K˘H
"
input/0˙˙˙˙˙˙˙˙˙
"
input/1˙˙˙˙˙˙˙˙˙
Ş "˘

0˙˙˙˙˙˙˙˙˙
__inference_call_15789344fJ˘G
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
0h
__inference_call_15789681K;˘8
1˘.

s1


s2

Ş "	
__inference_call_15789738cK˘H
A˘>

s1˙˙˙˙˙˙˙˙˙

s2˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙i
__inference_call_15789788L+˘(
!˘

x˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙Y
__inference_call_15789797<#˘ 
˘

x

Ş "§
E__inference_dense_3_layer_call_and_return_conditional_losses_15789818^0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙ô8
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 
*__inference_dense_3_layer_call_fn_15789827Q0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙ô8
Ş "˙˙˙˙˙˙˙˙˙Â
N__inference_distance_layer_3_layer_call_and_return_conditional_losses_15789618pK˘H
A˘>

s1˙˙˙˙˙˙˙˙˙

s2˙˙˙˙˙˙˙˙˙
Ş "!˘

0˙˙˙˙˙˙˙˙˙
 
3__inference_distance_layer_3_layer_call_fn_15789624cK˘H
A˘>

s1˙˙˙˙˙˙˙˙˙

s2˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙İ
G__inference_dropout_3_layer_call_and_return_conditional_losses_15789750^4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 İ
G__inference_dropout_3_layer_call_and_return_conditional_losses_15789755^4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 
,__inference_dropout_3_layer_call_fn_15789760Q4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p
Ş "˙˙˙˙˙˙˙˙˙
,__inference_dropout_3_layer_call_fn_15789765Q4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p 
Ş "˙˙˙˙˙˙˙˙˙İ
G__inference_flatten_3_layer_call_and_return_conditional_losses_15789803^4˘1
*˘'
%"
inputs˙˙˙˙˙˙˙˙˙
Ş "&˘#

0˙˙˙˙˙˙˙˙˙ô8
 
,__inference_flatten_3_layer_call_fn_15789808Q4˘1
*˘'
%"
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙ô8à
E__inference_model_3_layer_call_and_return_conditional_losses_15788849f˘c
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
 à
E__inference_model_3_layer_call_and_return_conditional_losses_15788863f˘c
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
 Ŝ
E__inference_model_3_layer_call_and_return_conditional_losses_15789382d˘a
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
 Ŝ
E__inference_model_3_layer_call_and_return_conditional_losses_15789406d˘a
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
 Ŝ
E__inference_model_3_layer_call_and_return_conditional_losses_15789464d˘a
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
 Ŝ
E__inference_model_3_layer_call_and_return_conditional_losses_15789488d˘a
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
 ı
*__inference_model_3_layer_call_fn_15788888f˘c
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
0˙˙˙˙˙˙˙˙˙ı
*__inference_model_3_layer_call_fn_15788912f˘c
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
0˙˙˙˙˙˙˙˙˙·
*__inference_model_3_layer_call_fn_15789416d˘a
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
0˙˙˙˙˙˙˙˙˙·
*__inference_model_3_layer_call_fn_15789426d˘a
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
0˙˙˙˙˙˙˙˙˙·
*__inference_model_3_layer_call_fn_15789498d˘a
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
0˙˙˙˙˙˙˙˙˙·
*__inference_model_3_layer_call_fn_15789508d˘a
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
0˙˙˙˙˙˙˙˙˙³
V__inference_one_hot_encoding_layer_3_layer_call_and_return_conditional_losses_15789774Y+˘(
!˘

x˙˙˙˙˙˙˙˙˙
Ş "*˘'
 
0˙˙˙˙˙˙˙˙˙
 
;__inference_one_hot_encoding_layer_3_layer_call_fn_15789779L+˘(
!˘

x˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙µ
J__inference_sequential_3_layer_call_and_return_conditional_losses_15788696g9˘6
/˘,
"
input_4˙˙˙˙˙˙˙˙˙
p

 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 µ
J__inference_sequential_3_layer_call_and_return_conditional_losses_15788708g9˘6
/˘,
"
input_4˙˙˙˙˙˙˙˙˙
p 

 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 ´
J__inference_sequential_3_layer_call_and_return_conditional_losses_15789529f8˘5
.˘+
!
inputs˙˙˙˙˙˙˙˙˙
p

 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 ´
J__inference_sequential_3_layer_call_and_return_conditional_losses_15789543f8˘5
.˘+
!
inputs˙˙˙˙˙˙˙˙˙
p 

 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 
/__inference_sequential_3_layer_call_fn_15788730Z9˘6
/˘,
"
input_4˙˙˙˙˙˙˙˙˙
p

 
Ş "˙˙˙˙˙˙˙˙˙
/__inference_sequential_3_layer_call_fn_15788751Z9˘6
/˘,
"
input_4˙˙˙˙˙˙˙˙˙
p 

 
Ş "˙˙˙˙˙˙˙˙˙
/__inference_sequential_3_layer_call_fn_15789552Y8˘5
.˘+
!
inputs˙˙˙˙˙˙˙˙˙
p

 
Ş "˙˙˙˙˙˙˙˙˙
/__inference_sequential_3_layer_call_fn_15789561Y8˘5
.˘+
!
inputs˙˙˙˙˙˙˙˙˙
p 

 
Ş "˙˙˙˙˙˙˙˙˙à
M__inference_siamese_model_3_layer_call_and_return_conditional_losses_15789104^˘[
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
 à
M__inference_siamese_model_3_layer_call_and_return_conditional_losses_15789128^˘[
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
 à
M__inference_siamese_model_3_layer_call_and_return_conditional_losses_15789186^˘[
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
 à
M__inference_siamese_model_3_layer_call_and_return_conditional_losses_15789210^˘[
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
 ı
2__inference_siamese_model_3_layer_call_fn_15789138^˘[
T˘Q
K˘H
"
input_1˙˙˙˙˙˙˙˙˙
"
input_2˙˙˙˙˙˙˙˙˙
p
Ş "˘

0˙˙˙˙˙˙˙˙˙ı
2__inference_siamese_model_3_layer_call_fn_15789148^˘[
T˘Q
K˘H
"
input_1˙˙˙˙˙˙˙˙˙
"
input_2˙˙˙˙˙˙˙˙˙
p 
Ş "˘

0˙˙˙˙˙˙˙˙˙ı
2__inference_siamese_model_3_layer_call_fn_15789220^˘[
T˘Q
K˘H
"
input/0˙˙˙˙˙˙˙˙˙
"
input/1˙˙˙˙˙˙˙˙˙
p
Ş "˘

0˙˙˙˙˙˙˙˙˙ı
2__inference_siamese_model_3_layer_call_fn_15789230^˘[
T˘Q
K˘H
"
input/0˙˙˙˙˙˙˙˙˙
"
input/1˙˙˙˙˙˙˙˙˙
p 
Ş "˘

0˙˙˙˙˙˙˙˙˙Í
&__inference_signature_wrapper_15789066˘k˘h
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