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
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ô8*
shared_namedense_5/kernel
s
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel* 
_output_shapes
:
ô8*
dtype0
q
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_5/bias
j
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes	
:*
dtype0

Adam/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ô8*&
shared_nameAdam/dense_5/kernel/m

)Adam/dense_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/m* 
_output_shapes
:
ô8*
dtype0

Adam/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_5/bias/m
x
'Adam/dense_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ô8*&
shared_nameAdam/dense_5/kernel/v

)Adam/dense_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/v* 
_output_shapes
:
ô8*
dtype0

Adam/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_5/bias/v
x
'Adam/dense_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/v*
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
VARIABLE_VALUEdense_5/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense_5/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEAdam/dense_5/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense_5/bias/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_5/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense_5/bias/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
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
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2dense_5/kerneldense_5/bias*
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
&__inference_signature_wrapper_26096213
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ú
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenametotal/Read/ReadVariableOpcount/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOp)Adam/dense_5/kernel/m/Read/ReadVariableOp'Adam/dense_5/bias/m/Read/ReadVariableOp)Adam/dense_5/kernel/v/Read/ReadVariableOp'Adam/dense_5/bias/v/Read/ReadVariableOpConst*
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
!__inference__traced_save_26097037
ñ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenametotalcount	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratedense_5/kerneldense_5/biasAdam/dense_5/kernel/mAdam/dense_5/bias/mAdam/dense_5/kernel/vAdam/dense_5/bias/v*
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
$__inference__traced_restore_26097086ŜÑ
í

2__inference_siamese_model_5_layer_call_fn_26096377
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
M__inference_siamese_model_5_layer_call_and_return_conditional_losses_260961762
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
__inference_call_26096885
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
cond_false_26096868*"
output_shapes
:˙˙˙˙˙˙˙˙˙*%
then_branchR
cond_true_260968672
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

9
cond_true_26096867
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

H
,__inference_dropout_5_layer_call_fn_26096912

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
G__inference_dropout_5_layer_call_and_return_conditional_losses_260957722
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

0
__inference_call_26095667
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
Ĥ
?
cond_false_26096748
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
½
m
V__inference_one_hot_encoding_layer_5_layer_call_and_return_conditional_losses_26096921
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
Ğ
e
,__inference_dropout_5_layer_call_fn_26096907

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
G__inference_dropout_5_layer_call_and_return_conditional_losses_260957672
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
öG
×
M__inference_siamese_model_5_layer_call_and_return_conditional_losses_26096251
input_1
input_2?
;model_5_sequential_5_dense_5_matmul_readvariableop_resource@
<model_5_sequential_5_dense_5_biasadd_readvariableop_resource
identity˘3model_5/sequential_5/dense_5/BiasAdd/ReadVariableOp˘5model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOp˘2model_5/sequential_5/dense_5/MatMul/ReadVariableOp˘4model_5/sequential_5/dense_5/MatMul_1/ReadVariableOpĦ
,model_5/sequential_5/dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2.
,model_5/sequential_5/dropout_5/dropout/ConstÒ
*model_5/sequential_5/dropout_5/dropout/MulMulinput_15model_5/sequential_5/dropout_5/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2,
*model_5/sequential_5/dropout_5/dropout/Mul
,model_5/sequential_5/dropout_5/dropout/ShapeShapeinput_1*
T0*
_output_shapes
:2.
,model_5/sequential_5/dropout_5/dropout/Shape
Cmodel_5/sequential_5/dropout_5/dropout/random_uniform/RandomUniformRandomUniform5model_5/sequential_5/dropout_5/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02E
Cmodel_5/sequential_5/dropout_5/dropout/random_uniform/RandomUniform³
5model_5/sequential_5/dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5model_5/sequential_5/dropout_5/dropout/GreaterEqual/yğ
3model_5/sequential_5/dropout_5/dropout/GreaterEqualGreaterEqualLmodel_5/sequential_5/dropout_5/dropout/random_uniform/RandomUniform:output:0>model_5/sequential_5/dropout_5/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙25
3model_5/sequential_5/dropout_5/dropout/GreaterEqualŬ
+model_5/sequential_5/dropout_5/dropout/CastCast7model_5/sequential_5/dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2-
+model_5/sequential_5/dropout_5/dropout/Cast÷
,model_5/sequential_5/dropout_5/dropout/Mul_1Mul.model_5/sequential_5/dropout_5/dropout/Mul:z:0/model_5/sequential_5/dropout_5/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2.
,model_5/sequential_5/dropout_5/dropout/Mul_1?
=model_5/sequential_5/one_hot_encoding_layer_5/PartitionedCallPartitionedCall0model_5/sequential_5/dropout_5/dropout/Mul_1:z:0*
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
__inference_call_260956672?
=model_5/sequential_5/one_hot_encoding_layer_5/PartitionedCall
$model_5/sequential_5/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2&
$model_5/sequential_5/flatten_5/Const
&model_5/sequential_5/flatten_5/ReshapeReshapeFmodel_5/sequential_5/one_hot_encoding_layer_5/PartitionedCall:output:0-model_5/sequential_5/flatten_5/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82(
&model_5/sequential_5/flatten_5/Reshapeĉ
2model_5/sequential_5/dense_5/MatMul/ReadVariableOpReadVariableOp;model_5_sequential_5_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype024
2model_5/sequential_5/dense_5/MatMul/ReadVariableOpô
#model_5/sequential_5/dense_5/MatMulMatMul/model_5/sequential_5/flatten_5/Reshape:output:0:model_5/sequential_5/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#model_5/sequential_5/dense_5/MatMulä
3model_5/sequential_5/dense_5/BiasAdd/ReadVariableOpReadVariableOp<model_5_sequential_5_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype025
3model_5/sequential_5/dense_5/BiasAdd/ReadVariableOpö
$model_5/sequential_5/dense_5/BiasAddBiasAdd-model_5/sequential_5/dense_5/MatMul:product:0;model_5/sequential_5/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2&
$model_5/sequential_5/dense_5/BiasAdd?
.model_5/sequential_5/dropout_5/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?20
.model_5/sequential_5/dropout_5/dropout_1/ConstĜ
,model_5/sequential_5/dropout_5/dropout_1/MulMulinput_27model_5/sequential_5/dropout_5/dropout_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2.
,model_5/sequential_5/dropout_5/dropout_1/Mul
.model_5/sequential_5/dropout_5/dropout_1/ShapeShapeinput_2*
T0*
_output_shapes
:20
.model_5/sequential_5/dropout_5/dropout_1/Shape
Emodel_5/sequential_5/dropout_5/dropout_1/random_uniform/RandomUniformRandomUniform7model_5/sequential_5/dropout_5/dropout_1/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02G
Emodel_5/sequential_5/dropout_5/dropout_1/random_uniform/RandomUniform·
7model_5/sequential_5/dropout_5/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    29
7model_5/sequential_5/dropout_5/dropout_1/GreaterEqual/y?
5model_5/sequential_5/dropout_5/dropout_1/GreaterEqualGreaterEqualNmodel_5/sequential_5/dropout_5/dropout_1/random_uniform/RandomUniform:output:0@model_5/sequential_5/dropout_5/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙27
5model_5/sequential_5/dropout_5/dropout_1/GreaterEqual?
-model_5/sequential_5/dropout_5/dropout_1/CastCast9model_5/sequential_5/dropout_5/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2/
-model_5/sequential_5/dropout_5/dropout_1/Cast˙
.model_5/sequential_5/dropout_5/dropout_1/Mul_1Mul0model_5/sequential_5/dropout_5/dropout_1/Mul:z:01model_5/sequential_5/dropout_5/dropout_1/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙20
.model_5/sequential_5/dropout_5/dropout_1/Mul_1Ğ
?model_5/sequential_5/one_hot_encoding_layer_5/PartitionedCall_1PartitionedCall2model_5/sequential_5/dropout_5/dropout_1/Mul_1:z:0*
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
__inference_call_260956672A
?model_5/sequential_5/one_hot_encoding_layer_5/PartitionedCall_1Ħ
&model_5/sequential_5/flatten_5/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2(
&model_5/sequential_5/flatten_5/Const_1
(model_5/sequential_5/flatten_5/Reshape_1ReshapeHmodel_5/sequential_5/one_hot_encoding_layer_5/PartitionedCall_1:output:0/model_5/sequential_5/flatten_5/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82*
(model_5/sequential_5/flatten_5/Reshape_1ê
4model_5/sequential_5/dense_5/MatMul_1/ReadVariableOpReadVariableOp;model_5_sequential_5_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype026
4model_5/sequential_5/dense_5/MatMul_1/ReadVariableOpü
%model_5/sequential_5/dense_5/MatMul_1MatMul1model_5/sequential_5/flatten_5/Reshape_1:output:0<model_5/sequential_5/dense_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%model_5/sequential_5/dense_5/MatMul_1è
5model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOpReadVariableOp<model_5_sequential_5_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype027
5model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOpŝ
&model_5/sequential_5/dense_5/BiasAdd_1BiasAdd/model_5/sequential_5/dense_5/MatMul_1:product:0=model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2(
&model_5/sequential_5/dense_5/BiasAdd_1Ħ
(model_5/distance_layer_5/PartitionedCallPartitionedCall-model_5/sequential_5/dense_5/BiasAdd:output:0/model_5/sequential_5/dense_5/BiasAdd_1:output:0*
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
__inference_call_260957412*
(model_5/distance_layer_5/PartitionedCallÛ
IdentityIdentity1model_5/distance_layer_5/PartitionedCall:output:04^model_5/sequential_5/dense_5/BiasAdd/ReadVariableOp6^model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOp3^model_5/sequential_5/dense_5/MatMul/ReadVariableOp5^model_5/sequential_5/dense_5/MatMul_1/ReadVariableOp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2j
3model_5/sequential_5/dense_5/BiasAdd/ReadVariableOp3model_5/sequential_5/dense_5/BiasAdd/ReadVariableOp2n
5model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOp5model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOp2h
2model_5/sequential_5/dense_5/MatMul/ReadVariableOp2model_5/sequential_5/dense_5/MatMul/ReadVariableOp2l
4model_5/sequential_5/dense_5/MatMul_1/ReadVariableOp4model_5/sequential_5/dense_5/MatMul_1/ReadVariableOp:Q M
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

S
3__inference_distance_layer_5_layer_call_fn_26096771
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
N__inference_distance_layer_5_layer_call_and_return_conditional_losses_260959862
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
ß%
Ê
!__inference__traced_save_26097037
file_prefix$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop4
0savev2_adam_dense_5_kernel_m_read_readvariableop2
.savev2_adam_dense_5_bias_m_read_readvariableop4
0savev2_adam_dense_5_kernel_v_read_readvariableop2
.savev2_adam_dense_5_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0 savev2_total_read_readvariableop savev2_count_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop0savev2_adam_dense_5_kernel_m_read_readvariableop.savev2_adam_dense_5_bias_m_read_readvariableop0savev2_adam_dense_5_kernel_v_read_readvariableop.savev2_adam_dense_5_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
¤9
Ò
$__inference__traced_restore_26097086
file_prefix
assignvariableop_total
assignvariableop_1_count 
assignvariableop_2_adam_iter"
assignvariableop_3_adam_beta_1"
assignvariableop_4_adam_beta_2!
assignvariableop_5_adam_decay)
%assignvariableop_6_adam_learning_rate%
!assignvariableop_7_dense_5_kernel#
assignvariableop_8_dense_5_bias,
(assignvariableop_9_adam_dense_5_kernel_m+
'assignvariableop_10_adam_dense_5_bias_m-
)assignvariableop_11_adam_dense_5_kernel_v+
'assignvariableop_12_adam_dense_5_bias_v
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
AssignVariableOp_7AssignVariableOp!assignvariableop_7_dense_5_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¤
AssignVariableOp_8AssignVariableOpassignvariableop_8_dense_5_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9­
AssignVariableOp_9AssignVariableOp(assignvariableop_9_adam_dense_5_kernel_mIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ż
AssignVariableOp_10AssignVariableOp'assignvariableop_10_adam_dense_5_bias_mIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11ħ
AssignVariableOp_11AssignVariableOp)assignvariableop_11_adam_dense_5_kernel_vIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Ż
AssignVariableOp_12AssignVariableOp'assignvariableop_12_adam_dense_5_bias_vIdentity_12:output:0"/device:CPU:0*
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
Ŝ@
Ħ
E__inference_model_5_layer_call_and_return_conditional_losses_26096611
inputs_0
inputs_17
3sequential_5_dense_5_matmul_readvariableop_resource8
4sequential_5_dense_5_biasadd_readvariableop_resource
identity˘+sequential_5/dense_5/BiasAdd/ReadVariableOp˘-sequential_5/dense_5/BiasAdd_1/ReadVariableOp˘*sequential_5/dense_5/MatMul/ReadVariableOp˘,sequential_5/dense_5/MatMul_1/ReadVariableOp
$sequential_5/dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2&
$sequential_5/dropout_5/dropout/Constğ
"sequential_5/dropout_5/dropout/MulMulinputs_0-sequential_5/dropout_5/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2$
"sequential_5/dropout_5/dropout/Mul
$sequential_5/dropout_5/dropout/ShapeShapeinputs_0*
T0*
_output_shapes
:2&
$sequential_5/dropout_5/dropout/Shapeú
;sequential_5/dropout_5/dropout/random_uniform/RandomUniformRandomUniform-sequential_5/dropout_5/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02=
;sequential_5/dropout_5/dropout/random_uniform/RandomUniform£
-sequential_5/dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2/
-sequential_5/dropout_5/dropout/GreaterEqual/y
+sequential_5/dropout_5/dropout/GreaterEqualGreaterEqualDsequential_5/dropout_5/dropout/random_uniform/RandomUniform:output:06sequential_5/dropout_5/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2-
+sequential_5/dropout_5/dropout/GreaterEqualĊ
#sequential_5/dropout_5/dropout/CastCast/sequential_5/dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#sequential_5/dropout_5/dropout/Cast×
$sequential_5/dropout_5/dropout/Mul_1Mul&sequential_5/dropout_5/dropout/Mul:z:0'sequential_5/dropout_5/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2&
$sequential_5/dropout_5/dropout/Mul_1
5sequential_5/one_hot_encoding_layer_5/PartitionedCallPartitionedCall(sequential_5/dropout_5/dropout/Mul_1:z:0*
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
__inference_call_2609566727
5sequential_5/one_hot_encoding_layer_5/PartitionedCall
sequential_5/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2
sequential_5/flatten_5/Constċ
sequential_5/flatten_5/ReshapeReshape>sequential_5/one_hot_encoding_layer_5/PartitionedCall:output:0%sequential_5/flatten_5/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82 
sequential_5/flatten_5/ReshapeÎ
*sequential_5/dense_5/MatMul/ReadVariableOpReadVariableOp3sequential_5_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02,
*sequential_5/dense_5/MatMul/ReadVariableOpÔ
sequential_5/dense_5/MatMulMatMul'sequential_5/flatten_5/Reshape:output:02sequential_5/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_5/dense_5/MatMulÌ
+sequential_5/dense_5/BiasAdd/ReadVariableOpReadVariableOp4sequential_5_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+sequential_5/dense_5/BiasAdd/ReadVariableOpÖ
sequential_5/dense_5/BiasAddBiasAdd%sequential_5/dense_5/MatMul:product:03sequential_5/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_5/dense_5/BiasAdd
&sequential_5/dropout_5/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2(
&sequential_5/dropout_5/dropout_1/ConstÁ
$sequential_5/dropout_5/dropout_1/MulMulinputs_1/sequential_5/dropout_5/dropout_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2&
$sequential_5/dropout_5/dropout_1/Mul
&sequential_5/dropout_5/dropout_1/ShapeShapeinputs_1*
T0*
_output_shapes
:2(
&sequential_5/dropout_5/dropout_1/Shape
=sequential_5/dropout_5/dropout_1/random_uniform/RandomUniformRandomUniform/sequential_5/dropout_5/dropout_1/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02?
=sequential_5/dropout_5/dropout_1/random_uniform/RandomUniform§
/sequential_5/dropout_5/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    21
/sequential_5/dropout_5/dropout_1/GreaterEqual/y£
-sequential_5/dropout_5/dropout_1/GreaterEqualGreaterEqualFsequential_5/dropout_5/dropout_1/random_uniform/RandomUniform:output:08sequential_5/dropout_5/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2/
-sequential_5/dropout_5/dropout_1/GreaterEqualË
%sequential_5/dropout_5/dropout_1/CastCast1sequential_5/dropout_5/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%sequential_5/dropout_5/dropout_1/Castß
&sequential_5/dropout_5/dropout_1/Mul_1Mul(sequential_5/dropout_5/dropout_1/Mul:z:0)sequential_5/dropout_5/dropout_1/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2(
&sequential_5/dropout_5/dropout_1/Mul_1
7sequential_5/one_hot_encoding_layer_5/PartitionedCall_1PartitionedCall*sequential_5/dropout_5/dropout_1/Mul_1:z:0*
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
__inference_call_2609566729
7sequential_5/one_hot_encoding_layer_5/PartitionedCall_1
sequential_5/flatten_5/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2 
sequential_5/flatten_5/Const_1í
 sequential_5/flatten_5/Reshape_1Reshape@sequential_5/one_hot_encoding_layer_5/PartitionedCall_1:output:0'sequential_5/flatten_5/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82"
 sequential_5/flatten_5/Reshape_1Ò
,sequential_5/dense_5/MatMul_1/ReadVariableOpReadVariableOp3sequential_5_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02.
,sequential_5/dense_5/MatMul_1/ReadVariableOpÜ
sequential_5/dense_5/MatMul_1MatMul)sequential_5/flatten_5/Reshape_1:output:04sequential_5/dense_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_5/dense_5/MatMul_1?
-sequential_5/dense_5/BiasAdd_1/ReadVariableOpReadVariableOp4sequential_5_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-sequential_5/dense_5/BiasAdd_1/ReadVariableOpŜ
sequential_5/dense_5/BiasAdd_1BiasAdd'sequential_5/dense_5/MatMul_1:product:05sequential_5/dense_5/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
sequential_5/dense_5/BiasAdd_1
 distance_layer_5/PartitionedCallPartitionedCall%sequential_5/dense_5/BiasAdd:output:0'sequential_5/dense_5/BiasAdd_1:output:0*
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
__inference_call_260957412"
 distance_layer_5/PartitionedCall³
IdentityIdentity)distance_layer_5/PartitionedCall:output:0,^sequential_5/dense_5/BiasAdd/ReadVariableOp.^sequential_5/dense_5/BiasAdd_1/ReadVariableOp+^sequential_5/dense_5/MatMul/ReadVariableOp-^sequential_5/dense_5/MatMul_1/ReadVariableOp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2Z
+sequential_5/dense_5/BiasAdd/ReadVariableOp+sequential_5/dense_5/BiasAdd/ReadVariableOp2^
-sequential_5/dense_5/BiasAdd_1/ReadVariableOp-sequential_5/dense_5/BiasAdd_1/ReadVariableOp2X
*sequential_5/dense_5/MatMul/ReadVariableOp*sequential_5/dense_5/MatMul/ReadVariableOp2\
,sequential_5/dense_5/MatMul_1/ReadVariableOp,sequential_5/dense_5/MatMul_1/ReadVariableOp:R N
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
Ô@

E__inference_model_5_layer_call_and_return_conditional_losses_26096101

inputs
inputs_17
3sequential_5_dense_5_matmul_readvariableop_resource8
4sequential_5_dense_5_biasadd_readvariableop_resource
identity˘+sequential_5/dense_5/BiasAdd/ReadVariableOp˘-sequential_5/dense_5/BiasAdd_1/ReadVariableOp˘*sequential_5/dense_5/MatMul/ReadVariableOp˘,sequential_5/dense_5/MatMul_1/ReadVariableOp
$sequential_5/dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2&
$sequential_5/dropout_5/dropout/Constı
"sequential_5/dropout_5/dropout/MulMulinputs-sequential_5/dropout_5/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2$
"sequential_5/dropout_5/dropout/Mul
$sequential_5/dropout_5/dropout/ShapeShapeinputs*
T0*
_output_shapes
:2&
$sequential_5/dropout_5/dropout/Shapeú
;sequential_5/dropout_5/dropout/random_uniform/RandomUniformRandomUniform-sequential_5/dropout_5/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02=
;sequential_5/dropout_5/dropout/random_uniform/RandomUniform£
-sequential_5/dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2/
-sequential_5/dropout_5/dropout/GreaterEqual/y
+sequential_5/dropout_5/dropout/GreaterEqualGreaterEqualDsequential_5/dropout_5/dropout/random_uniform/RandomUniform:output:06sequential_5/dropout_5/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2-
+sequential_5/dropout_5/dropout/GreaterEqualĊ
#sequential_5/dropout_5/dropout/CastCast/sequential_5/dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#sequential_5/dropout_5/dropout/Cast×
$sequential_5/dropout_5/dropout/Mul_1Mul&sequential_5/dropout_5/dropout/Mul:z:0'sequential_5/dropout_5/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2&
$sequential_5/dropout_5/dropout/Mul_1
5sequential_5/one_hot_encoding_layer_5/PartitionedCallPartitionedCall(sequential_5/dropout_5/dropout/Mul_1:z:0*
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
__inference_call_2609566727
5sequential_5/one_hot_encoding_layer_5/PartitionedCall
sequential_5/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2
sequential_5/flatten_5/Constċ
sequential_5/flatten_5/ReshapeReshape>sequential_5/one_hot_encoding_layer_5/PartitionedCall:output:0%sequential_5/flatten_5/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82 
sequential_5/flatten_5/ReshapeÎ
*sequential_5/dense_5/MatMul/ReadVariableOpReadVariableOp3sequential_5_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02,
*sequential_5/dense_5/MatMul/ReadVariableOpÔ
sequential_5/dense_5/MatMulMatMul'sequential_5/flatten_5/Reshape:output:02sequential_5/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_5/dense_5/MatMulÌ
+sequential_5/dense_5/BiasAdd/ReadVariableOpReadVariableOp4sequential_5_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+sequential_5/dense_5/BiasAdd/ReadVariableOpÖ
sequential_5/dense_5/BiasAddBiasAdd%sequential_5/dense_5/MatMul:product:03sequential_5/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_5/dense_5/BiasAdd
&sequential_5/dropout_5/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2(
&sequential_5/dropout_5/dropout_1/ConstÁ
$sequential_5/dropout_5/dropout_1/MulMulinputs_1/sequential_5/dropout_5/dropout_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2&
$sequential_5/dropout_5/dropout_1/Mul
&sequential_5/dropout_5/dropout_1/ShapeShapeinputs_1*
T0*
_output_shapes
:2(
&sequential_5/dropout_5/dropout_1/Shape
=sequential_5/dropout_5/dropout_1/random_uniform/RandomUniformRandomUniform/sequential_5/dropout_5/dropout_1/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02?
=sequential_5/dropout_5/dropout_1/random_uniform/RandomUniform§
/sequential_5/dropout_5/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    21
/sequential_5/dropout_5/dropout_1/GreaterEqual/y£
-sequential_5/dropout_5/dropout_1/GreaterEqualGreaterEqualFsequential_5/dropout_5/dropout_1/random_uniform/RandomUniform:output:08sequential_5/dropout_5/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2/
-sequential_5/dropout_5/dropout_1/GreaterEqualË
%sequential_5/dropout_5/dropout_1/CastCast1sequential_5/dropout_5/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%sequential_5/dropout_5/dropout_1/Castß
&sequential_5/dropout_5/dropout_1/Mul_1Mul(sequential_5/dropout_5/dropout_1/Mul:z:0)sequential_5/dropout_5/dropout_1/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2(
&sequential_5/dropout_5/dropout_1/Mul_1
7sequential_5/one_hot_encoding_layer_5/PartitionedCall_1PartitionedCall*sequential_5/dropout_5/dropout_1/Mul_1:z:0*
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
__inference_call_2609566729
7sequential_5/one_hot_encoding_layer_5/PartitionedCall_1
sequential_5/flatten_5/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2 
sequential_5/flatten_5/Const_1í
 sequential_5/flatten_5/Reshape_1Reshape@sequential_5/one_hot_encoding_layer_5/PartitionedCall_1:output:0'sequential_5/flatten_5/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82"
 sequential_5/flatten_5/Reshape_1Ò
,sequential_5/dense_5/MatMul_1/ReadVariableOpReadVariableOp3sequential_5_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02.
,sequential_5/dense_5/MatMul_1/ReadVariableOpÜ
sequential_5/dense_5/MatMul_1MatMul)sequential_5/flatten_5/Reshape_1:output:04sequential_5/dense_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_5/dense_5/MatMul_1?
-sequential_5/dense_5/BiasAdd_1/ReadVariableOpReadVariableOp4sequential_5_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-sequential_5/dense_5/BiasAdd_1/ReadVariableOpŜ
sequential_5/dense_5/BiasAdd_1BiasAdd'sequential_5/dense_5/MatMul_1:product:05sequential_5/dense_5/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
sequential_5/dense_5/BiasAdd_1
 distance_layer_5/PartitionedCallPartitionedCall%sequential_5/dense_5/BiasAdd:output:0'sequential_5/dense_5/BiasAdd_1:output:0*
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
__inference_call_260957412"
 distance_layer_5/PartitionedCall³
IdentityIdentity)distance_layer_5/PartitionedCall:output:0,^sequential_5/dense_5/BiasAdd/ReadVariableOp.^sequential_5/dense_5/BiasAdd_1/ReadVariableOp+^sequential_5/dense_5/MatMul/ReadVariableOp-^sequential_5/dense_5/MatMul_1/ReadVariableOp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2Z
+sequential_5/dense_5/BiasAdd/ReadVariableOp+sequential_5/dense_5/BiasAdd/ReadVariableOp2^
-sequential_5/dense_5/BiasAdd_1/ReadVariableOp-sequential_5/dense_5/BiasAdd_1/ReadVariableOp2X
*sequential_5/dense_5/MatMul/ReadVariableOp*sequential_5/dense_5/MatMul/ReadVariableOp2\
,sequential_5/dense_5/MatMul_1/ReadVariableOp,sequential_5/dense_5/MatMul_1/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:PL
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
â
9
cond_true_26096470
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
"
9
__inference_call_26095741
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
cond_false_26095724*"
output_shapes
:˙˙˙˙˙˙˙˙˙*%
then_branchR
cond_true_260957232
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
í

2__inference_siamese_model_5_layer_call_fn_26096285
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
M__inference_siamese_model_5_layer_call_and_return_conditional_losses_260961762
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
Ċ
Ü
J__inference_sequential_5_layer_call_and_return_conditional_losses_26095843
input_6
dense_5_26095837
dense_5_26095839
identity˘dense_5/StatefulPartitionedCall˘!dropout_5/StatefulPartitionedCallö
!dropout_5/StatefulPartitionedCallStatefulPartitionedCallinput_6*
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
G__inference_dropout_5_layer_call_and_return_conditional_losses_260957672#
!dropout_5/StatefulPartitionedCall²
(one_hot_encoding_layer_5/PartitionedCallPartitionedCall*dropout_5/StatefulPartitionedCall:output:0*
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
V__inference_one_hot_encoding_layer_5_layer_call_and_return_conditional_losses_260957942*
(one_hot_encoding_layer_5/PartitionedCall
flatten_5/PartitionedCallPartitionedCall1one_hot_encoding_layer_5/PartitionedCall:output:0*
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
G__inference_flatten_5_layer_call_and_return_conditional_losses_260958082
flatten_5/PartitionedCallµ
dense_5/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0dense_5_26095837dense_5_26095839*
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
E__inference_dense_5_layer_call_and_return_conditional_losses_260958262!
dense_5/StatefulPartitionedCall?
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0 ^dense_5/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall:Q M
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_6
	
Ŝ
E__inference_dense_5_layer_call_and_return_conditional_losses_26096965

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
?
Ĉ
M__inference_siamese_model_5_layer_call_and_return_conditional_losses_26096176	
input
input_1
model_5_26096170
model_5_26096172
identity˘model_5/StatefulPartitionedCall
model_5/StatefulPartitionedCallStatefulPartitionedCallinputinput_1model_5_26096170model_5_26096172*
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
E__inference_model_5_layer_call_and_return_conditional_losses_260961252!
model_5/StatefulPartitionedCall
IdentityIdentity(model_5/StatefulPartitionedCall:output:0 ^model_5/StatefulPartitionedCall*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2B
model_5/StatefulPartitionedCallmodel_5/StatefulPartitionedCall:O K
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameinput:OK
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameinput
¨
ĝ
E__inference_model_5_layer_call_and_return_conditional_losses_26096028

inputs
inputs_1
sequential_5_26096018
sequential_5_26096020
identity˘$sequential_5/StatefulPartitionedCall˘&sequential_5/StatefulPartitionedCall_1²
$sequential_5/StatefulPartitionedCallStatefulPartitionedCallinputssequential_5_26096018sequential_5_26096020*
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
J__inference_sequential_5_layer_call_and_return_conditional_losses_260958702&
$sequential_5/StatefulPartitionedCall¸
&sequential_5/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1sequential_5_26096018sequential_5_26096020*
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
J__inference_sequential_5_layer_call_and_return_conditional_losses_260958702(
&sequential_5/StatefulPartitionedCall_1Ĉ
 distance_layer_5/PartitionedCallPartitionedCall-sequential_5/StatefulPartitionedCall:output:0/sequential_5/StatefulPartitionedCall_1:output:0*
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
N__inference_distance_layer_5_layer_call_and_return_conditional_losses_260959862"
 distance_layer_5/PartitionedCallÉ
IdentityIdentity)distance_layer_5/PartitionedCall:output:0%^sequential_5/StatefulPartitionedCall'^sequential_5/StatefulPartitionedCall_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2L
$sequential_5/StatefulPartitionedCall$sequential_5/StatefulPartitionedCall2P
&sequential_5/StatefulPartitionedCall_1&sequential_5/StatefulPartitionedCall_1:P L
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
*__inference_model_5_layer_call_fn_26096645
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
E__inference_model_5_layer_call_and_return_conditional_losses_260960282
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
·

&__inference_signature_wrapper_26096213
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
#__inference__wrapped_model_260957512
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
Ú
ĥ
#__inference__wrapped_model_26095751
input_1
input_2
siamese_model_5_26095745
siamese_model_5_26095747
identity˘'siamese_model_5/StatefulPartitionedCall
'siamese_model_5/StatefulPartitionedCallStatefulPartitionedCallinput_1input_2siamese_model_5_26095745siamese_model_5_26095747*
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
__inference_call_260957442)
'siamese_model_5/StatefulPartitionedCallŞ
IdentityIdentity0siamese_model_5/StatefulPartitionedCall:output:0(^siamese_model_5/StatefulPartitionedCall*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2R
'siamese_model_5/StatefulPartitionedCall'siamese_model_5/StatefulPartitionedCall:Q M
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
ó

/__inference_sequential_5_layer_call_fn_26095877
input_6
unknown
	unknown_0
identity˘StatefulPartitionedCall˙
StatefulPartitionedCallStatefulPartitionedCallinput_6unknown	unknown_0*
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
J__inference_sequential_5_layer_call_and_return_conditional_losses_260958702
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
_user_specified_name	input_6

?
cond_false_26096471
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
Ĝ)
Ħ
E__inference_model_5_layer_call_and_return_conditional_losses_26096635
inputs_0
inputs_17
3sequential_5_dense_5_matmul_readvariableop_resource8
4sequential_5_dense_5_biasadd_readvariableop_resource
identity˘+sequential_5/dense_5/BiasAdd/ReadVariableOp˘-sequential_5/dense_5/BiasAdd_1/ReadVariableOp˘*sequential_5/dense_5/MatMul/ReadVariableOp˘,sequential_5/dense_5/MatMul_1/ReadVariableOp
sequential_5/dropout_5/IdentityIdentityinputs_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2!
sequential_5/dropout_5/Identity
5sequential_5/one_hot_encoding_layer_5/PartitionedCallPartitionedCall(sequential_5/dropout_5/Identity:output:0*
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
__inference_call_2609566727
5sequential_5/one_hot_encoding_layer_5/PartitionedCall
sequential_5/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2
sequential_5/flatten_5/Constċ
sequential_5/flatten_5/ReshapeReshape>sequential_5/one_hot_encoding_layer_5/PartitionedCall:output:0%sequential_5/flatten_5/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82 
sequential_5/flatten_5/ReshapeÎ
*sequential_5/dense_5/MatMul/ReadVariableOpReadVariableOp3sequential_5_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02,
*sequential_5/dense_5/MatMul/ReadVariableOpÔ
sequential_5/dense_5/MatMulMatMul'sequential_5/flatten_5/Reshape:output:02sequential_5/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_5/dense_5/MatMulÌ
+sequential_5/dense_5/BiasAdd/ReadVariableOpReadVariableOp4sequential_5_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+sequential_5/dense_5/BiasAdd/ReadVariableOpÖ
sequential_5/dense_5/BiasAddBiasAdd%sequential_5/dense_5/MatMul:product:03sequential_5/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_5/dense_5/BiasAdd
!sequential_5/dropout_5/Identity_1Identityinputs_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2#
!sequential_5/dropout_5/Identity_1
7sequential_5/one_hot_encoding_layer_5/PartitionedCall_1PartitionedCall*sequential_5/dropout_5/Identity_1:output:0*
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
__inference_call_2609566729
7sequential_5/one_hot_encoding_layer_5/PartitionedCall_1
sequential_5/flatten_5/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2 
sequential_5/flatten_5/Const_1í
 sequential_5/flatten_5/Reshape_1Reshape@sequential_5/one_hot_encoding_layer_5/PartitionedCall_1:output:0'sequential_5/flatten_5/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82"
 sequential_5/flatten_5/Reshape_1Ò
,sequential_5/dense_5/MatMul_1/ReadVariableOpReadVariableOp3sequential_5_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02.
,sequential_5/dense_5/MatMul_1/ReadVariableOpÜ
sequential_5/dense_5/MatMul_1MatMul)sequential_5/flatten_5/Reshape_1:output:04sequential_5/dense_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_5/dense_5/MatMul_1?
-sequential_5/dense_5/BiasAdd_1/ReadVariableOpReadVariableOp4sequential_5_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-sequential_5/dense_5/BiasAdd_1/ReadVariableOpŜ
sequential_5/dense_5/BiasAdd_1BiasAdd'sequential_5/dense_5/MatMul_1:product:05sequential_5/dense_5/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
sequential_5/dense_5/BiasAdd_1
 distance_layer_5/PartitionedCallPartitionedCall%sequential_5/dense_5/BiasAdd:output:0'sequential_5/dense_5/BiasAdd_1:output:0*
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
__inference_call_260957412"
 distance_layer_5/PartitionedCall³
IdentityIdentity)distance_layer_5/PartitionedCall:output:0,^sequential_5/dense_5/BiasAdd/ReadVariableOp.^sequential_5/dense_5/BiasAdd_1/ReadVariableOp+^sequential_5/dense_5/MatMul/ReadVariableOp-^sequential_5/dense_5/MatMul_1/ReadVariableOp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2Z
+sequential_5/dense_5/BiasAdd/ReadVariableOp+sequential_5/dense_5/BiasAdd/ReadVariableOp2^
-sequential_5/dense_5/BiasAdd_1/ReadVariableOp-sequential_5/dense_5/BiasAdd_1/ReadVariableOp2X
*sequential_5/dense_5/MatMul/ReadVariableOp*sequential_5/dense_5/MatMul/ReadVariableOp2\
,sequential_5/dense_5/MatMul_1/ReadVariableOp,sequential_5/dense_5/MatMul_1/ReadVariableOp:R N
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
?

/__inference_sequential_5_layer_call_fn_26096708

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
J__inference_sequential_5_layer_call_and_return_conditional_losses_260958912
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

9
cond_true_26095968
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
ċ

*__inference_dense_5_layer_call_fn_26096974

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
E__inference_dense_5_layer_call_and_return_conditional_losses_260958262
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
Â
Û
J__inference_sequential_5_layer_call_and_return_conditional_losses_26095870

inputs
dense_5_26095864
dense_5_26095866
identity˘dense_5/StatefulPartitionedCall˘!dropout_5/StatefulPartitionedCallġ
!dropout_5/StatefulPartitionedCallStatefulPartitionedCallinputs*
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
G__inference_dropout_5_layer_call_and_return_conditional_losses_260957672#
!dropout_5/StatefulPartitionedCall²
(one_hot_encoding_layer_5/PartitionedCallPartitionedCall*dropout_5/StatefulPartitionedCall:output:0*
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
V__inference_one_hot_encoding_layer_5_layer_call_and_return_conditional_losses_260957942*
(one_hot_encoding_layer_5/PartitionedCall
flatten_5/PartitionedCallPartitionedCall1one_hot_encoding_layer_5/PartitionedCall:output:0*
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
G__inference_flatten_5_layer_call_and_return_conditional_losses_260958082
flatten_5/PartitionedCallµ
dense_5/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0dense_5_26095864dense_5_26095866*
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
E__inference_dense_5_layer_call_and_return_conditional_losses_260958262!
dense_5/StatefulPartitionedCall?
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0 ^dense_5/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
î-
£
__inference_call_26096401
input_0
input_1?
;model_5_sequential_5_dense_5_matmul_readvariableop_resource@
<model_5_sequential_5_dense_5_biasadd_readvariableop_resource
identity˘3model_5/sequential_5/dense_5/BiasAdd/ReadVariableOp˘5model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOp˘2model_5/sequential_5/dense_5/MatMul/ReadVariableOp˘4model_5/sequential_5/dense_5/MatMul_1/ReadVariableOp
'model_5/sequential_5/dropout_5/IdentityIdentityinput_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2)
'model_5/sequential_5/dropout_5/Identity?
=model_5/sequential_5/one_hot_encoding_layer_5/PartitionedCallPartitionedCall0model_5/sequential_5/dropout_5/Identity:output:0*
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
__inference_call_260956672?
=model_5/sequential_5/one_hot_encoding_layer_5/PartitionedCall
$model_5/sequential_5/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2&
$model_5/sequential_5/flatten_5/Const
&model_5/sequential_5/flatten_5/ReshapeReshapeFmodel_5/sequential_5/one_hot_encoding_layer_5/PartitionedCall:output:0-model_5/sequential_5/flatten_5/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82(
&model_5/sequential_5/flatten_5/Reshapeĉ
2model_5/sequential_5/dense_5/MatMul/ReadVariableOpReadVariableOp;model_5_sequential_5_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype024
2model_5/sequential_5/dense_5/MatMul/ReadVariableOpô
#model_5/sequential_5/dense_5/MatMulMatMul/model_5/sequential_5/flatten_5/Reshape:output:0:model_5/sequential_5/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#model_5/sequential_5/dense_5/MatMulä
3model_5/sequential_5/dense_5/BiasAdd/ReadVariableOpReadVariableOp<model_5_sequential_5_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype025
3model_5/sequential_5/dense_5/BiasAdd/ReadVariableOpö
$model_5/sequential_5/dense_5/BiasAddBiasAdd-model_5/sequential_5/dense_5/MatMul:product:0;model_5/sequential_5/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2&
$model_5/sequential_5/dense_5/BiasAdd
)model_5/sequential_5/dropout_5/Identity_1Identityinput_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2+
)model_5/sequential_5/dropout_5/Identity_1Ğ
?model_5/sequential_5/one_hot_encoding_layer_5/PartitionedCall_1PartitionedCall2model_5/sequential_5/dropout_5/Identity_1:output:0*
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
__inference_call_260956672A
?model_5/sequential_5/one_hot_encoding_layer_5/PartitionedCall_1Ħ
&model_5/sequential_5/flatten_5/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2(
&model_5/sequential_5/flatten_5/Const_1
(model_5/sequential_5/flatten_5/Reshape_1ReshapeHmodel_5/sequential_5/one_hot_encoding_layer_5/PartitionedCall_1:output:0/model_5/sequential_5/flatten_5/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82*
(model_5/sequential_5/flatten_5/Reshape_1ê
4model_5/sequential_5/dense_5/MatMul_1/ReadVariableOpReadVariableOp;model_5_sequential_5_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype026
4model_5/sequential_5/dense_5/MatMul_1/ReadVariableOpü
%model_5/sequential_5/dense_5/MatMul_1MatMul1model_5/sequential_5/flatten_5/Reshape_1:output:0<model_5/sequential_5/dense_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%model_5/sequential_5/dense_5/MatMul_1è
5model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOpReadVariableOp<model_5_sequential_5_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype027
5model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOpŝ
&model_5/sequential_5/dense_5/BiasAdd_1BiasAdd/model_5/sequential_5/dense_5/MatMul_1:product:0=model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2(
&model_5/sequential_5/dense_5/BiasAdd_1Ħ
(model_5/distance_layer_5/PartitionedCallPartitionedCall-model_5/sequential_5/dense_5/BiasAdd:output:0/model_5/sequential_5/dense_5/BiasAdd_1:output:0*
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
__inference_call_260957412*
(model_5/distance_layer_5/PartitionedCallÛ
IdentityIdentity1model_5/distance_layer_5/PartitionedCall:output:04^model_5/sequential_5/dense_5/BiasAdd/ReadVariableOp6^model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOp3^model_5/sequential_5/dense_5/MatMul/ReadVariableOp5^model_5/sequential_5/dense_5/MatMul_1/ReadVariableOp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2j
3model_5/sequential_5/dense_5/BiasAdd/ReadVariableOp3model_5/sequential_5/dense_5/BiasAdd/ReadVariableOp2n
5model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOp5model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOp2h
2model_5/sequential_5/dense_5/MatMul/ReadVariableOp2model_5/sequential_5/dense_5/MatMul/ReadVariableOp2l
4model_5/sequential_5/dense_5/MatMul_1/ReadVariableOp4model_5/sequential_5/dense_5/MatMul_1/ReadVariableOp:Q M
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
î,
£
__inference_call_26096491
input_0
input_1?
;model_5_sequential_5_dense_5_matmul_readvariableop_resource@
<model_5_sequential_5_dense_5_biasadd_readvariableop_resource
identity˘3model_5/sequential_5/dense_5/BiasAdd/ReadVariableOp˘5model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOp˘2model_5/sequential_5/dense_5/MatMul/ReadVariableOp˘4model_5/sequential_5/dense_5/MatMul_1/ReadVariableOp
'model_5/sequential_5/dropout_5/IdentityIdentityinput_0*
T0* 
_output_shapes
:
2)
'model_5/sequential_5/dropout_5/Identity
=model_5/sequential_5/one_hot_encoding_layer_5/PartitionedCallPartitionedCall0model_5/sequential_5/dropout_5/Identity:output:0*
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
__inference_call_260964142?
=model_5/sequential_5/one_hot_encoding_layer_5/PartitionedCall
$model_5/sequential_5/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2&
$model_5/sequential_5/flatten_5/Constŭ
&model_5/sequential_5/flatten_5/ReshapeReshapeFmodel_5/sequential_5/one_hot_encoding_layer_5/PartitionedCall:output:0-model_5/sequential_5/flatten_5/Const:output:0*
T0* 
_output_shapes
:
ô82(
&model_5/sequential_5/flatten_5/Reshapeĉ
2model_5/sequential_5/dense_5/MatMul/ReadVariableOpReadVariableOp;model_5_sequential_5_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype024
2model_5/sequential_5/dense_5/MatMul/ReadVariableOpì
#model_5/sequential_5/dense_5/MatMulMatMul/model_5/sequential_5/flatten_5/Reshape:output:0:model_5/sequential_5/dense_5/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2%
#model_5/sequential_5/dense_5/MatMulä
3model_5/sequential_5/dense_5/BiasAdd/ReadVariableOpReadVariableOp<model_5_sequential_5_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype025
3model_5/sequential_5/dense_5/BiasAdd/ReadVariableOpî
$model_5/sequential_5/dense_5/BiasAddBiasAdd-model_5/sequential_5/dense_5/MatMul:product:0;model_5/sequential_5/dense_5/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2&
$model_5/sequential_5/dense_5/BiasAdd
)model_5/sequential_5/dropout_5/Identity_1Identityinput_1*
T0* 
_output_shapes
:
2+
)model_5/sequential_5/dropout_5/Identity_1£
?model_5/sequential_5/one_hot_encoding_layer_5/PartitionedCall_1PartitionedCall2model_5/sequential_5/dropout_5/Identity_1:output:0*
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
__inference_call_260964142A
?model_5/sequential_5/one_hot_encoding_layer_5/PartitionedCall_1Ħ
&model_5/sequential_5/flatten_5/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2(
&model_5/sequential_5/flatten_5/Const_1
(model_5/sequential_5/flatten_5/Reshape_1ReshapeHmodel_5/sequential_5/one_hot_encoding_layer_5/PartitionedCall_1:output:0/model_5/sequential_5/flatten_5/Const_1:output:0*
T0* 
_output_shapes
:
ô82*
(model_5/sequential_5/flatten_5/Reshape_1ê
4model_5/sequential_5/dense_5/MatMul_1/ReadVariableOpReadVariableOp;model_5_sequential_5_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype026
4model_5/sequential_5/dense_5/MatMul_1/ReadVariableOpô
%model_5/sequential_5/dense_5/MatMul_1MatMul1model_5/sequential_5/flatten_5/Reshape_1:output:0<model_5/sequential_5/dense_5/MatMul_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2'
%model_5/sequential_5/dense_5/MatMul_1è
5model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOpReadVariableOp<model_5_sequential_5_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype027
5model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOpö
&model_5/sequential_5/dense_5/BiasAdd_1BiasAdd/model_5/sequential_5/dense_5/MatMul_1:product:0=model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2(
&model_5/sequential_5/dense_5/BiasAdd_1
(model_5/distance_layer_5/PartitionedCallPartitionedCall-model_5/sequential_5/dense_5/BiasAdd:output:0/model_5/sequential_5/dense_5/BiasAdd_1:output:0*
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
__inference_call_260964882*
(model_5/distance_layer_5/PartitionedCallÓ
IdentityIdentity1model_5/distance_layer_5/PartitionedCall:output:04^model_5/sequential_5/dense_5/BiasAdd/ReadVariableOp6^model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOp3^model_5/sequential_5/dense_5/MatMul/ReadVariableOp5^model_5/sequential_5/dense_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes	
:2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :
:
::2j
3model_5/sequential_5/dense_5/BiasAdd/ReadVariableOp3model_5/sequential_5/dense_5/BiasAdd/ReadVariableOp2n
5model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOp5model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOp2h
2model_5/sequential_5/dense_5/MatMul/ReadVariableOp2model_5/sequential_5/dense_5/MatMul/ReadVariableOp2l
4model_5/sequential_5/dense_5/MatMul_1/ReadVariableOp4model_5/sequential_5/dense_5/MatMul_1/ReadVariableOp:I E
 
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
ĥ
ü
E__inference_model_5_layer_call_and_return_conditional_losses_26096010
	sequence1
	sequence2
sequential_5_26096000
sequential_5_26096002
identity˘$sequential_5/StatefulPartitionedCall˘&sequential_5/StatefulPartitionedCall_1µ
$sequential_5/StatefulPartitionedCallStatefulPartitionedCall	sequence1sequential_5_26096000sequential_5_26096002*
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
J__inference_sequential_5_layer_call_and_return_conditional_losses_260958912&
$sequential_5/StatefulPartitionedCallı
&sequential_5/StatefulPartitionedCall_1StatefulPartitionedCall	sequence2sequential_5_26096000sequential_5_26096002*
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
J__inference_sequential_5_layer_call_and_return_conditional_losses_260958912(
&sequential_5/StatefulPartitionedCall_1Ĉ
 distance_layer_5/PartitionedCallPartitionedCall-sequential_5/StatefulPartitionedCall:output:0/sequential_5/StatefulPartitionedCall_1:output:0*
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
N__inference_distance_layer_5_layer_call_and_return_conditional_losses_260959862"
 distance_layer_5/PartitionedCallÉ
IdentityIdentity)distance_layer_5/PartitionedCall:output:0%^sequential_5/StatefulPartitionedCall'^sequential_5/StatefulPartitionedCall_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2L
$sequential_5/StatefulPartitionedCall$sequential_5/StatefulPartitionedCall2P
&sequential_5/StatefulPartitionedCall_1&sequential_5/StatefulPartitionedCall_1:S O
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
Ĝ
0
__inference_call_26096414
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

f
G__inference_dropout_5_layer_call_and_return_conditional_losses_26095767

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
ı
c
G__inference_flatten_5_layer_call_and_return_conditional_losses_26095808

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
Î
e
G__inference_dropout_5_layer_call_and_return_conditional_losses_26095772

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
Ĥ
?
cond_false_26095724
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
ĥ
ü
E__inference_model_5_layer_call_and_return_conditional_losses_26095996
	sequence1
	sequence2
sequential_5_26095921
sequential_5_26095923
identity˘$sequential_5/StatefulPartitionedCall˘&sequential_5/StatefulPartitionedCall_1µ
$sequential_5/StatefulPartitionedCallStatefulPartitionedCall	sequence1sequential_5_26095921sequential_5_26095923*
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
J__inference_sequential_5_layer_call_and_return_conditional_losses_260958702&
$sequential_5/StatefulPartitionedCallı
&sequential_5/StatefulPartitionedCall_1StatefulPartitionedCall	sequence2sequential_5_26095921sequential_5_26095923*
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
J__inference_sequential_5_layer_call_and_return_conditional_losses_260958702(
&sequential_5/StatefulPartitionedCall_1Ĉ
 distance_layer_5/PartitionedCallPartitionedCall-sequential_5/StatefulPartitionedCall:output:0/sequential_5/StatefulPartitionedCall_1:output:0*
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
N__inference_distance_layer_5_layer_call_and_return_conditional_losses_260959862"
 distance_layer_5/PartitionedCallÉ
IdentityIdentity)distance_layer_5/PartitionedCall:output:0%^sequential_5/StatefulPartitionedCall'^sequential_5/StatefulPartitionedCall_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2L
$sequential_5/StatefulPartitionedCall$sequential_5/StatefulPartitionedCall2P
&sequential_5/StatefulPartitionedCall_1&sequential_5/StatefulPartitionedCall_1:S O
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
*__inference_model_5_layer_call_fn_26096059
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
E__inference_model_5_layer_call_and_return_conditional_losses_260960522
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
M__inference_siamese_model_5_layer_call_and_return_conditional_losses_26096357
input_0
input_1?
;model_5_sequential_5_dense_5_matmul_readvariableop_resource@
<model_5_sequential_5_dense_5_biasadd_readvariableop_resource
identity˘3model_5/sequential_5/dense_5/BiasAdd/ReadVariableOp˘5model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOp˘2model_5/sequential_5/dense_5/MatMul/ReadVariableOp˘4model_5/sequential_5/dense_5/MatMul_1/ReadVariableOp
'model_5/sequential_5/dropout_5/IdentityIdentityinput_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2)
'model_5/sequential_5/dropout_5/Identity?
=model_5/sequential_5/one_hot_encoding_layer_5/PartitionedCallPartitionedCall0model_5/sequential_5/dropout_5/Identity:output:0*
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
__inference_call_260956672?
=model_5/sequential_5/one_hot_encoding_layer_5/PartitionedCall
$model_5/sequential_5/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2&
$model_5/sequential_5/flatten_5/Const
&model_5/sequential_5/flatten_5/ReshapeReshapeFmodel_5/sequential_5/one_hot_encoding_layer_5/PartitionedCall:output:0-model_5/sequential_5/flatten_5/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82(
&model_5/sequential_5/flatten_5/Reshapeĉ
2model_5/sequential_5/dense_5/MatMul/ReadVariableOpReadVariableOp;model_5_sequential_5_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype024
2model_5/sequential_5/dense_5/MatMul/ReadVariableOpô
#model_5/sequential_5/dense_5/MatMulMatMul/model_5/sequential_5/flatten_5/Reshape:output:0:model_5/sequential_5/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#model_5/sequential_5/dense_5/MatMulä
3model_5/sequential_5/dense_5/BiasAdd/ReadVariableOpReadVariableOp<model_5_sequential_5_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype025
3model_5/sequential_5/dense_5/BiasAdd/ReadVariableOpö
$model_5/sequential_5/dense_5/BiasAddBiasAdd-model_5/sequential_5/dense_5/MatMul:product:0;model_5/sequential_5/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2&
$model_5/sequential_5/dense_5/BiasAdd
)model_5/sequential_5/dropout_5/Identity_1Identityinput_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2+
)model_5/sequential_5/dropout_5/Identity_1Ğ
?model_5/sequential_5/one_hot_encoding_layer_5/PartitionedCall_1PartitionedCall2model_5/sequential_5/dropout_5/Identity_1:output:0*
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
__inference_call_260956672A
?model_5/sequential_5/one_hot_encoding_layer_5/PartitionedCall_1Ħ
&model_5/sequential_5/flatten_5/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2(
&model_5/sequential_5/flatten_5/Const_1
(model_5/sequential_5/flatten_5/Reshape_1ReshapeHmodel_5/sequential_5/one_hot_encoding_layer_5/PartitionedCall_1:output:0/model_5/sequential_5/flatten_5/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82*
(model_5/sequential_5/flatten_5/Reshape_1ê
4model_5/sequential_5/dense_5/MatMul_1/ReadVariableOpReadVariableOp;model_5_sequential_5_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype026
4model_5/sequential_5/dense_5/MatMul_1/ReadVariableOpü
%model_5/sequential_5/dense_5/MatMul_1MatMul1model_5/sequential_5/flatten_5/Reshape_1:output:0<model_5/sequential_5/dense_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%model_5/sequential_5/dense_5/MatMul_1è
5model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOpReadVariableOp<model_5_sequential_5_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype027
5model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOpŝ
&model_5/sequential_5/dense_5/BiasAdd_1BiasAdd/model_5/sequential_5/dense_5/MatMul_1:product:0=model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2(
&model_5/sequential_5/dense_5/BiasAdd_1Ħ
(model_5/distance_layer_5/PartitionedCallPartitionedCall-model_5/sequential_5/dense_5/BiasAdd:output:0/model_5/sequential_5/dense_5/BiasAdd_1:output:0*
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
__inference_call_260957412*
(model_5/distance_layer_5/PartitionedCallÛ
IdentityIdentity1model_5/distance_layer_5/PartitionedCall:output:04^model_5/sequential_5/dense_5/BiasAdd/ReadVariableOp6^model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOp3^model_5/sequential_5/dense_5/MatMul/ReadVariableOp5^model_5/sequential_5/dense_5/MatMul_1/ReadVariableOp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2j
3model_5/sequential_5/dense_5/BiasAdd/ReadVariableOp3model_5/sequential_5/dense_5/BiasAdd/ReadVariableOp2n
5model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOp5model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOp2h
2model_5/sequential_5/dense_5/MatMul/ReadVariableOp2model_5/sequential_5/dense_5/MatMul/ReadVariableOp2l
4model_5/sequential_5/dense_5/MatMul_1/ReadVariableOp4model_5/sequential_5/dense_5/MatMul_1/ReadVariableOp:Q M
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
cond_true_26095723
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

¸
J__inference_sequential_5_layer_call_and_return_conditional_losses_26095855
input_6
dense_5_26095849
dense_5_26095851
identity˘dense_5/StatefulPartitionedCallŜ
dropout_5/PartitionedCallPartitionedCallinput_6*
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
G__inference_dropout_5_layer_call_and_return_conditional_losses_260957722
dropout_5/PartitionedCallŞ
(one_hot_encoding_layer_5/PartitionedCallPartitionedCall"dropout_5/PartitionedCall:output:0*
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
V__inference_one_hot_encoding_layer_5_layer_call_and_return_conditional_losses_260957942*
(one_hot_encoding_layer_5/PartitionedCall
flatten_5/PartitionedCallPartitionedCall1one_hot_encoding_layer_5/PartitionedCall:output:0*
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
G__inference_flatten_5_layer_call_and_return_conditional_losses_260958082
flatten_5/PartitionedCallµ
dense_5/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0dense_5_26095849dense_5_26095851*
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
E__inference_dense_5_layer_call_and_return_conditional_losses_260958262!
dense_5/StatefulPartitionedCall
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0 ^dense_5/StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:Q M
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_6
ĝ

J__inference_sequential_5_layer_call_and_return_conditional_losses_26096690

inputs*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identity˘dense_5/BiasAdd/ReadVariableOp˘dense_5/MatMul/ReadVariableOpo
dropout_5/IdentityIdentityinputs*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout_5/Identityĉ
(one_hot_encoding_layer_5/PartitionedCallPartitionedCalldropout_5/Identity:output:0*
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
__inference_call_260956672*
(one_hot_encoding_layer_5/PartitionedCalls
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2
flatten_5/Constħ
flatten_5/ReshapeReshape1one_hot_encoding_layer_5/PartitionedCall:output:0flatten_5/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82
flatten_5/Reshape§
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02
dense_5/MatMul/ReadVariableOp 
dense_5/MatMulMatMulflatten_5/Reshape:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp˘
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_5/BiasAdd?
IdentityIdentitydense_5/BiasAdd:output:0^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ô"
n
N__inference_distance_layer_5_layer_call_and_return_conditional_losses_26096765
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
cond_false_26096748*"
output_shapes
:˙˙˙˙˙˙˙˙˙*%
then_branchR
cond_true_260967472
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
ı
c
G__inference_flatten_5_layer_call_and_return_conditional_losses_26096950

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
í

2__inference_siamese_model_5_layer_call_fn_26096295
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
M__inference_siamese_model_5_layer_call_and_return_conditional_losses_260961762
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

·
J__inference_sequential_5_layer_call_and_return_conditional_losses_26095891

inputs
dense_5_26095885
dense_5_26095887
identity˘dense_5/StatefulPartitionedCallŬ
dropout_5/PartitionedCallPartitionedCallinputs*
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
G__inference_dropout_5_layer_call_and_return_conditional_losses_260957722
dropout_5/PartitionedCallŞ
(one_hot_encoding_layer_5/PartitionedCallPartitionedCall"dropout_5/PartitionedCall:output:0*
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
V__inference_one_hot_encoding_layer_5_layer_call_and_return_conditional_losses_260957942*
(one_hot_encoding_layer_5/PartitionedCall
flatten_5/PartitionedCallPartitionedCall1one_hot_encoding_layer_5/PartitionedCall:output:0*
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
G__inference_flatten_5_layer_call_and_return_conditional_losses_260958082
flatten_5/PartitionedCallµ
dense_5/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0dense_5_26095885dense_5_26095887*
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
E__inference_dense_5_layer_call_and_return_conditional_losses_260958262!
dense_5/StatefulPartitionedCall
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0 ^dense_5/StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ŝ@
Ħ
E__inference_model_5_layer_call_and_return_conditional_losses_26096529
inputs_0
inputs_17
3sequential_5_dense_5_matmul_readvariableop_resource8
4sequential_5_dense_5_biasadd_readvariableop_resource
identity˘+sequential_5/dense_5/BiasAdd/ReadVariableOp˘-sequential_5/dense_5/BiasAdd_1/ReadVariableOp˘*sequential_5/dense_5/MatMul/ReadVariableOp˘,sequential_5/dense_5/MatMul_1/ReadVariableOp
$sequential_5/dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2&
$sequential_5/dropout_5/dropout/Constğ
"sequential_5/dropout_5/dropout/MulMulinputs_0-sequential_5/dropout_5/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2$
"sequential_5/dropout_5/dropout/Mul
$sequential_5/dropout_5/dropout/ShapeShapeinputs_0*
T0*
_output_shapes
:2&
$sequential_5/dropout_5/dropout/Shapeú
;sequential_5/dropout_5/dropout/random_uniform/RandomUniformRandomUniform-sequential_5/dropout_5/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02=
;sequential_5/dropout_5/dropout/random_uniform/RandomUniform£
-sequential_5/dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2/
-sequential_5/dropout_5/dropout/GreaterEqual/y
+sequential_5/dropout_5/dropout/GreaterEqualGreaterEqualDsequential_5/dropout_5/dropout/random_uniform/RandomUniform:output:06sequential_5/dropout_5/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2-
+sequential_5/dropout_5/dropout/GreaterEqualĊ
#sequential_5/dropout_5/dropout/CastCast/sequential_5/dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#sequential_5/dropout_5/dropout/Cast×
$sequential_5/dropout_5/dropout/Mul_1Mul&sequential_5/dropout_5/dropout/Mul:z:0'sequential_5/dropout_5/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2&
$sequential_5/dropout_5/dropout/Mul_1
5sequential_5/one_hot_encoding_layer_5/PartitionedCallPartitionedCall(sequential_5/dropout_5/dropout/Mul_1:z:0*
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
__inference_call_2609566727
5sequential_5/one_hot_encoding_layer_5/PartitionedCall
sequential_5/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2
sequential_5/flatten_5/Constċ
sequential_5/flatten_5/ReshapeReshape>sequential_5/one_hot_encoding_layer_5/PartitionedCall:output:0%sequential_5/flatten_5/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82 
sequential_5/flatten_5/ReshapeÎ
*sequential_5/dense_5/MatMul/ReadVariableOpReadVariableOp3sequential_5_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02,
*sequential_5/dense_5/MatMul/ReadVariableOpÔ
sequential_5/dense_5/MatMulMatMul'sequential_5/flatten_5/Reshape:output:02sequential_5/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_5/dense_5/MatMulÌ
+sequential_5/dense_5/BiasAdd/ReadVariableOpReadVariableOp4sequential_5_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+sequential_5/dense_5/BiasAdd/ReadVariableOpÖ
sequential_5/dense_5/BiasAddBiasAdd%sequential_5/dense_5/MatMul:product:03sequential_5/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_5/dense_5/BiasAdd
&sequential_5/dropout_5/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2(
&sequential_5/dropout_5/dropout_1/ConstÁ
$sequential_5/dropout_5/dropout_1/MulMulinputs_1/sequential_5/dropout_5/dropout_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2&
$sequential_5/dropout_5/dropout_1/Mul
&sequential_5/dropout_5/dropout_1/ShapeShapeinputs_1*
T0*
_output_shapes
:2(
&sequential_5/dropout_5/dropout_1/Shape
=sequential_5/dropout_5/dropout_1/random_uniform/RandomUniformRandomUniform/sequential_5/dropout_5/dropout_1/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02?
=sequential_5/dropout_5/dropout_1/random_uniform/RandomUniform§
/sequential_5/dropout_5/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    21
/sequential_5/dropout_5/dropout_1/GreaterEqual/y£
-sequential_5/dropout_5/dropout_1/GreaterEqualGreaterEqualFsequential_5/dropout_5/dropout_1/random_uniform/RandomUniform:output:08sequential_5/dropout_5/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2/
-sequential_5/dropout_5/dropout_1/GreaterEqualË
%sequential_5/dropout_5/dropout_1/CastCast1sequential_5/dropout_5/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%sequential_5/dropout_5/dropout_1/Castß
&sequential_5/dropout_5/dropout_1/Mul_1Mul(sequential_5/dropout_5/dropout_1/Mul:z:0)sequential_5/dropout_5/dropout_1/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2(
&sequential_5/dropout_5/dropout_1/Mul_1
7sequential_5/one_hot_encoding_layer_5/PartitionedCall_1PartitionedCall*sequential_5/dropout_5/dropout_1/Mul_1:z:0*
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
__inference_call_2609566729
7sequential_5/one_hot_encoding_layer_5/PartitionedCall_1
sequential_5/flatten_5/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2 
sequential_5/flatten_5/Const_1í
 sequential_5/flatten_5/Reshape_1Reshape@sequential_5/one_hot_encoding_layer_5/PartitionedCall_1:output:0'sequential_5/flatten_5/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82"
 sequential_5/flatten_5/Reshape_1Ò
,sequential_5/dense_5/MatMul_1/ReadVariableOpReadVariableOp3sequential_5_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02.
,sequential_5/dense_5/MatMul_1/ReadVariableOpÜ
sequential_5/dense_5/MatMul_1MatMul)sequential_5/flatten_5/Reshape_1:output:04sequential_5/dense_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_5/dense_5/MatMul_1?
-sequential_5/dense_5/BiasAdd_1/ReadVariableOpReadVariableOp4sequential_5_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-sequential_5/dense_5/BiasAdd_1/ReadVariableOpŜ
sequential_5/dense_5/BiasAdd_1BiasAdd'sequential_5/dense_5/MatMul_1:product:05sequential_5/dense_5/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
sequential_5/dense_5/BiasAdd_1
 distance_layer_5/PartitionedCallPartitionedCall%sequential_5/dense_5/BiasAdd:output:0'sequential_5/dense_5/BiasAdd_1:output:0*
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
__inference_call_260957412"
 distance_layer_5/PartitionedCall³
IdentityIdentity)distance_layer_5/PartitionedCall:output:0,^sequential_5/dense_5/BiasAdd/ReadVariableOp.^sequential_5/dense_5/BiasAdd_1/ReadVariableOp+^sequential_5/dense_5/MatMul/ReadVariableOp-^sequential_5/dense_5/MatMul_1/ReadVariableOp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2Z
+sequential_5/dense_5/BiasAdd/ReadVariableOp+sequential_5/dense_5/BiasAdd/ReadVariableOp2^
-sequential_5/dense_5/BiasAdd_1/ReadVariableOp-sequential_5/dense_5/BiasAdd_1/ReadVariableOp2X
*sequential_5/dense_5/MatMul/ReadVariableOp*sequential_5/dense_5/MatMul/ReadVariableOp2\
,sequential_5/dense_5/MatMul_1/ReadVariableOp,sequential_5/dense_5/MatMul_1/ReadVariableOp:R N
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
§
H
,__inference_flatten_5_layer_call_fn_26096955

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
G__inference_flatten_5_layer_call_and_return_conditional_losses_260958082
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
¨
ĝ
E__inference_model_5_layer_call_and_return_conditional_losses_26096052

inputs
inputs_1
sequential_5_26096042
sequential_5_26096044
identity˘$sequential_5/StatefulPartitionedCall˘&sequential_5/StatefulPartitionedCall_1²
$sequential_5/StatefulPartitionedCallStatefulPartitionedCallinputssequential_5_26096042sequential_5_26096044*
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
J__inference_sequential_5_layer_call_and_return_conditional_losses_260958912&
$sequential_5/StatefulPartitionedCall¸
&sequential_5/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1sequential_5_26096042sequential_5_26096044*
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
J__inference_sequential_5_layer_call_and_return_conditional_losses_260958912(
&sequential_5/StatefulPartitionedCall_1Ĉ
 distance_layer_5/PartitionedCallPartitionedCall-sequential_5/StatefulPartitionedCall:output:0/sequential_5/StatefulPartitionedCall_1:output:0*
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
N__inference_distance_layer_5_layer_call_and_return_conditional_losses_260959862"
 distance_layer_5/PartitionedCallÉ
IdentityIdentity)distance_layer_5/PartitionedCall:output:0%^sequential_5/StatefulPartitionedCall'^sequential_5/StatefulPartitionedCall_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2L
$sequential_5/StatefulPartitionedCall$sequential_5/StatefulPartitionedCall2P
&sequential_5/StatefulPartitionedCall_1&sequential_5/StatefulPartitionedCall_1:P L
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
cond_false_26095969
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
Ĥ
?
cond_false_26096868
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
	
Ŝ
E__inference_dense_5_layer_call_and_return_conditional_losses_26095826

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
ĥ
R
;__inference_one_hot_encoding_layer_5_layer_call_fn_26096926
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
V__inference_one_hot_encoding_layer_5_layer_call_and_return_conditional_losses_260957942
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
â
9
cond_true_26096810
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
öG
×
M__inference_siamese_model_5_layer_call_and_return_conditional_losses_26096333
input_0
input_1?
;model_5_sequential_5_dense_5_matmul_readvariableop_resource@
<model_5_sequential_5_dense_5_biasadd_readvariableop_resource
identity˘3model_5/sequential_5/dense_5/BiasAdd/ReadVariableOp˘5model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOp˘2model_5/sequential_5/dense_5/MatMul/ReadVariableOp˘4model_5/sequential_5/dense_5/MatMul_1/ReadVariableOpĦ
,model_5/sequential_5/dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2.
,model_5/sequential_5/dropout_5/dropout/ConstÒ
*model_5/sequential_5/dropout_5/dropout/MulMulinput_05model_5/sequential_5/dropout_5/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2,
*model_5/sequential_5/dropout_5/dropout/Mul
,model_5/sequential_5/dropout_5/dropout/ShapeShapeinput_0*
T0*
_output_shapes
:2.
,model_5/sequential_5/dropout_5/dropout/Shape
Cmodel_5/sequential_5/dropout_5/dropout/random_uniform/RandomUniformRandomUniform5model_5/sequential_5/dropout_5/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02E
Cmodel_5/sequential_5/dropout_5/dropout/random_uniform/RandomUniform³
5model_5/sequential_5/dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5model_5/sequential_5/dropout_5/dropout/GreaterEqual/yğ
3model_5/sequential_5/dropout_5/dropout/GreaterEqualGreaterEqualLmodel_5/sequential_5/dropout_5/dropout/random_uniform/RandomUniform:output:0>model_5/sequential_5/dropout_5/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙25
3model_5/sequential_5/dropout_5/dropout/GreaterEqualŬ
+model_5/sequential_5/dropout_5/dropout/CastCast7model_5/sequential_5/dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2-
+model_5/sequential_5/dropout_5/dropout/Cast÷
,model_5/sequential_5/dropout_5/dropout/Mul_1Mul.model_5/sequential_5/dropout_5/dropout/Mul:z:0/model_5/sequential_5/dropout_5/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2.
,model_5/sequential_5/dropout_5/dropout/Mul_1?
=model_5/sequential_5/one_hot_encoding_layer_5/PartitionedCallPartitionedCall0model_5/sequential_5/dropout_5/dropout/Mul_1:z:0*
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
__inference_call_260956672?
=model_5/sequential_5/one_hot_encoding_layer_5/PartitionedCall
$model_5/sequential_5/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2&
$model_5/sequential_5/flatten_5/Const
&model_5/sequential_5/flatten_5/ReshapeReshapeFmodel_5/sequential_5/one_hot_encoding_layer_5/PartitionedCall:output:0-model_5/sequential_5/flatten_5/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82(
&model_5/sequential_5/flatten_5/Reshapeĉ
2model_5/sequential_5/dense_5/MatMul/ReadVariableOpReadVariableOp;model_5_sequential_5_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype024
2model_5/sequential_5/dense_5/MatMul/ReadVariableOpô
#model_5/sequential_5/dense_5/MatMulMatMul/model_5/sequential_5/flatten_5/Reshape:output:0:model_5/sequential_5/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#model_5/sequential_5/dense_5/MatMulä
3model_5/sequential_5/dense_5/BiasAdd/ReadVariableOpReadVariableOp<model_5_sequential_5_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype025
3model_5/sequential_5/dense_5/BiasAdd/ReadVariableOpö
$model_5/sequential_5/dense_5/BiasAddBiasAdd-model_5/sequential_5/dense_5/MatMul:product:0;model_5/sequential_5/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2&
$model_5/sequential_5/dense_5/BiasAdd?
.model_5/sequential_5/dropout_5/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?20
.model_5/sequential_5/dropout_5/dropout_1/ConstĜ
,model_5/sequential_5/dropout_5/dropout_1/MulMulinput_17model_5/sequential_5/dropout_5/dropout_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2.
,model_5/sequential_5/dropout_5/dropout_1/Mul
.model_5/sequential_5/dropout_5/dropout_1/ShapeShapeinput_1*
T0*
_output_shapes
:20
.model_5/sequential_5/dropout_5/dropout_1/Shape
Emodel_5/sequential_5/dropout_5/dropout_1/random_uniform/RandomUniformRandomUniform7model_5/sequential_5/dropout_5/dropout_1/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02G
Emodel_5/sequential_5/dropout_5/dropout_1/random_uniform/RandomUniform·
7model_5/sequential_5/dropout_5/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    29
7model_5/sequential_5/dropout_5/dropout_1/GreaterEqual/y?
5model_5/sequential_5/dropout_5/dropout_1/GreaterEqualGreaterEqualNmodel_5/sequential_5/dropout_5/dropout_1/random_uniform/RandomUniform:output:0@model_5/sequential_5/dropout_5/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙27
5model_5/sequential_5/dropout_5/dropout_1/GreaterEqual?
-model_5/sequential_5/dropout_5/dropout_1/CastCast9model_5/sequential_5/dropout_5/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2/
-model_5/sequential_5/dropout_5/dropout_1/Cast˙
.model_5/sequential_5/dropout_5/dropout_1/Mul_1Mul0model_5/sequential_5/dropout_5/dropout_1/Mul:z:01model_5/sequential_5/dropout_5/dropout_1/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙20
.model_5/sequential_5/dropout_5/dropout_1/Mul_1Ğ
?model_5/sequential_5/one_hot_encoding_layer_5/PartitionedCall_1PartitionedCall2model_5/sequential_5/dropout_5/dropout_1/Mul_1:z:0*
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
__inference_call_260956672A
?model_5/sequential_5/one_hot_encoding_layer_5/PartitionedCall_1Ħ
&model_5/sequential_5/flatten_5/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2(
&model_5/sequential_5/flatten_5/Const_1
(model_5/sequential_5/flatten_5/Reshape_1ReshapeHmodel_5/sequential_5/one_hot_encoding_layer_5/PartitionedCall_1:output:0/model_5/sequential_5/flatten_5/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82*
(model_5/sequential_5/flatten_5/Reshape_1ê
4model_5/sequential_5/dense_5/MatMul_1/ReadVariableOpReadVariableOp;model_5_sequential_5_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype026
4model_5/sequential_5/dense_5/MatMul_1/ReadVariableOpü
%model_5/sequential_5/dense_5/MatMul_1MatMul1model_5/sequential_5/flatten_5/Reshape_1:output:0<model_5/sequential_5/dense_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%model_5/sequential_5/dense_5/MatMul_1è
5model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOpReadVariableOp<model_5_sequential_5_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype027
5model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOpŝ
&model_5/sequential_5/dense_5/BiasAdd_1BiasAdd/model_5/sequential_5/dense_5/MatMul_1:product:0=model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2(
&model_5/sequential_5/dense_5/BiasAdd_1Ħ
(model_5/distance_layer_5/PartitionedCallPartitionedCall-model_5/sequential_5/dense_5/BiasAdd:output:0/model_5/sequential_5/dense_5/BiasAdd_1:output:0*
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
__inference_call_260957412*
(model_5/distance_layer_5/PartitionedCallÛ
IdentityIdentity1model_5/distance_layer_5/PartitionedCall:output:04^model_5/sequential_5/dense_5/BiasAdd/ReadVariableOp6^model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOp3^model_5/sequential_5/dense_5/MatMul/ReadVariableOp5^model_5/sequential_5/dense_5/MatMul_1/ReadVariableOp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2j
3model_5/sequential_5/dense_5/BiasAdd/ReadVariableOp3model_5/sequential_5/dense_5/BiasAdd/ReadVariableOp2n
5model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOp5model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOp2h
2model_5/sequential_5/dense_5/MatMul/ReadVariableOp2model_5/sequential_5/dense_5/MatMul/ReadVariableOp2l
4model_5/sequential_5/dense_5/MatMul_1/ReadVariableOp4model_5/sequential_5/dense_5/MatMul_1/ReadVariableOp:Q M
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
Î
e
G__inference_dropout_5_layer_call_and_return_conditional_losses_26096902

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
Ĝ
0
__inference_call_26096944
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
Ĝ)
Ħ
E__inference_model_5_layer_call_and_return_conditional_losses_26096553
inputs_0
inputs_17
3sequential_5_dense_5_matmul_readvariableop_resource8
4sequential_5_dense_5_biasadd_readvariableop_resource
identity˘+sequential_5/dense_5/BiasAdd/ReadVariableOp˘-sequential_5/dense_5/BiasAdd_1/ReadVariableOp˘*sequential_5/dense_5/MatMul/ReadVariableOp˘,sequential_5/dense_5/MatMul_1/ReadVariableOp
sequential_5/dropout_5/IdentityIdentityinputs_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2!
sequential_5/dropout_5/Identity
5sequential_5/one_hot_encoding_layer_5/PartitionedCallPartitionedCall(sequential_5/dropout_5/Identity:output:0*
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
__inference_call_2609566727
5sequential_5/one_hot_encoding_layer_5/PartitionedCall
sequential_5/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2
sequential_5/flatten_5/Constċ
sequential_5/flatten_5/ReshapeReshape>sequential_5/one_hot_encoding_layer_5/PartitionedCall:output:0%sequential_5/flatten_5/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82 
sequential_5/flatten_5/ReshapeÎ
*sequential_5/dense_5/MatMul/ReadVariableOpReadVariableOp3sequential_5_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02,
*sequential_5/dense_5/MatMul/ReadVariableOpÔ
sequential_5/dense_5/MatMulMatMul'sequential_5/flatten_5/Reshape:output:02sequential_5/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_5/dense_5/MatMulÌ
+sequential_5/dense_5/BiasAdd/ReadVariableOpReadVariableOp4sequential_5_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+sequential_5/dense_5/BiasAdd/ReadVariableOpÖ
sequential_5/dense_5/BiasAddBiasAdd%sequential_5/dense_5/MatMul:product:03sequential_5/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_5/dense_5/BiasAdd
!sequential_5/dropout_5/Identity_1Identityinputs_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2#
!sequential_5/dropout_5/Identity_1
7sequential_5/one_hot_encoding_layer_5/PartitionedCall_1PartitionedCall*sequential_5/dropout_5/Identity_1:output:0*
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
__inference_call_2609566729
7sequential_5/one_hot_encoding_layer_5/PartitionedCall_1
sequential_5/flatten_5/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2 
sequential_5/flatten_5/Const_1í
 sequential_5/flatten_5/Reshape_1Reshape@sequential_5/one_hot_encoding_layer_5/PartitionedCall_1:output:0'sequential_5/flatten_5/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82"
 sequential_5/flatten_5/Reshape_1Ò
,sequential_5/dense_5/MatMul_1/ReadVariableOpReadVariableOp3sequential_5_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02.
,sequential_5/dense_5/MatMul_1/ReadVariableOpÜ
sequential_5/dense_5/MatMul_1MatMul)sequential_5/flatten_5/Reshape_1:output:04sequential_5/dense_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_5/dense_5/MatMul_1?
-sequential_5/dense_5/BiasAdd_1/ReadVariableOpReadVariableOp4sequential_5_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-sequential_5/dense_5/BiasAdd_1/ReadVariableOpŜ
sequential_5/dense_5/BiasAdd_1BiasAdd'sequential_5/dense_5/MatMul_1:product:05sequential_5/dense_5/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
sequential_5/dense_5/BiasAdd_1
 distance_layer_5/PartitionedCallPartitionedCall%sequential_5/dense_5/BiasAdd:output:0'sequential_5/dense_5/BiasAdd_1:output:0*
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
__inference_call_260957412"
 distance_layer_5/PartitionedCall³
IdentityIdentity)distance_layer_5/PartitionedCall:output:0,^sequential_5/dense_5/BiasAdd/ReadVariableOp.^sequential_5/dense_5/BiasAdd_1/ReadVariableOp+^sequential_5/dense_5/MatMul/ReadVariableOp-^sequential_5/dense_5/MatMul_1/ReadVariableOp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2Z
+sequential_5/dense_5/BiasAdd/ReadVariableOp+sequential_5/dense_5/BiasAdd/ReadVariableOp2^
-sequential_5/dense_5/BiasAdd_1/ReadVariableOp-sequential_5/dense_5/BiasAdd_1/ReadVariableOp2X
*sequential_5/dense_5/MatMul/ReadVariableOp*sequential_5/dense_5/MatMul/ReadVariableOp2\
,sequential_5/dense_5/MatMul_1/ReadVariableOp,sequential_5/dense_5/MatMul_1/ReadVariableOp:R N
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

f
G__inference_dropout_5_layer_call_and_return_conditional_losses_26096897

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
?)

E__inference_model_5_layer_call_and_return_conditional_losses_26096125

inputs
inputs_17
3sequential_5_dense_5_matmul_readvariableop_resource8
4sequential_5_dense_5_biasadd_readvariableop_resource
identity˘+sequential_5/dense_5/BiasAdd/ReadVariableOp˘-sequential_5/dense_5/BiasAdd_1/ReadVariableOp˘*sequential_5/dense_5/MatMul/ReadVariableOp˘,sequential_5/dense_5/MatMul_1/ReadVariableOp
sequential_5/dropout_5/IdentityIdentityinputs*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2!
sequential_5/dropout_5/Identity
5sequential_5/one_hot_encoding_layer_5/PartitionedCallPartitionedCall(sequential_5/dropout_5/Identity:output:0*
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
__inference_call_2609566727
5sequential_5/one_hot_encoding_layer_5/PartitionedCall
sequential_5/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2
sequential_5/flatten_5/Constċ
sequential_5/flatten_5/ReshapeReshape>sequential_5/one_hot_encoding_layer_5/PartitionedCall:output:0%sequential_5/flatten_5/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82 
sequential_5/flatten_5/ReshapeÎ
*sequential_5/dense_5/MatMul/ReadVariableOpReadVariableOp3sequential_5_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02,
*sequential_5/dense_5/MatMul/ReadVariableOpÔ
sequential_5/dense_5/MatMulMatMul'sequential_5/flatten_5/Reshape:output:02sequential_5/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_5/dense_5/MatMulÌ
+sequential_5/dense_5/BiasAdd/ReadVariableOpReadVariableOp4sequential_5_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+sequential_5/dense_5/BiasAdd/ReadVariableOpÖ
sequential_5/dense_5/BiasAddBiasAdd%sequential_5/dense_5/MatMul:product:03sequential_5/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_5/dense_5/BiasAdd
!sequential_5/dropout_5/Identity_1Identityinputs_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2#
!sequential_5/dropout_5/Identity_1
7sequential_5/one_hot_encoding_layer_5/PartitionedCall_1PartitionedCall*sequential_5/dropout_5/Identity_1:output:0*
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
__inference_call_2609566729
7sequential_5/one_hot_encoding_layer_5/PartitionedCall_1
sequential_5/flatten_5/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2 
sequential_5/flatten_5/Const_1í
 sequential_5/flatten_5/Reshape_1Reshape@sequential_5/one_hot_encoding_layer_5/PartitionedCall_1:output:0'sequential_5/flatten_5/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82"
 sequential_5/flatten_5/Reshape_1Ò
,sequential_5/dense_5/MatMul_1/ReadVariableOpReadVariableOp3sequential_5_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02.
,sequential_5/dense_5/MatMul_1/ReadVariableOpÜ
sequential_5/dense_5/MatMul_1MatMul)sequential_5/flatten_5/Reshape_1:output:04sequential_5/dense_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_5/dense_5/MatMul_1?
-sequential_5/dense_5/BiasAdd_1/ReadVariableOpReadVariableOp4sequential_5_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-sequential_5/dense_5/BiasAdd_1/ReadVariableOpŜ
sequential_5/dense_5/BiasAdd_1BiasAdd'sequential_5/dense_5/MatMul_1:product:05sequential_5/dense_5/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
sequential_5/dense_5/BiasAdd_1
 distance_layer_5/PartitionedCallPartitionedCall%sequential_5/dense_5/BiasAdd:output:0'sequential_5/dense_5/BiasAdd_1:output:0*
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
__inference_call_260957412"
 distance_layer_5/PartitionedCall³
IdentityIdentity)distance_layer_5/PartitionedCall:output:0,^sequential_5/dense_5/BiasAdd/ReadVariableOp.^sequential_5/dense_5/BiasAdd_1/ReadVariableOp+^sequential_5/dense_5/MatMul/ReadVariableOp-^sequential_5/dense_5/MatMul_1/ReadVariableOp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2Z
+sequential_5/dense_5/BiasAdd/ReadVariableOp+sequential_5/dense_5/BiasAdd/ReadVariableOp2^
-sequential_5/dense_5/BiasAdd_1/ReadVariableOp-sequential_5/dense_5/BiasAdd_1/ReadVariableOp2X
*sequential_5/dense_5/MatMul/ReadVariableOp*sequential_5/dense_5/MatMul/ReadVariableOp2\
,sequential_5/dense_5/MatMul_1/ReadVariableOp,sequential_5/dense_5/MatMul_1/ReadVariableOp:P L
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
*__inference_model_5_layer_call_fn_26096035
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
E__inference_model_5_layer_call_and_return_conditional_losses_260960282
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
?

*__inference_model_5_layer_call_fn_26096563
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
E__inference_model_5_layer_call_and_return_conditional_losses_260961012
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

?
cond_false_26096811
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

0
__inference_call_26096935
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
˘.
×
M__inference_siamese_model_5_layer_call_and_return_conditional_losses_26096275
input_1
input_2?
;model_5_sequential_5_dense_5_matmul_readvariableop_resource@
<model_5_sequential_5_dense_5_biasadd_readvariableop_resource
identity˘3model_5/sequential_5/dense_5/BiasAdd/ReadVariableOp˘5model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOp˘2model_5/sequential_5/dense_5/MatMul/ReadVariableOp˘4model_5/sequential_5/dense_5/MatMul_1/ReadVariableOp
'model_5/sequential_5/dropout_5/IdentityIdentityinput_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2)
'model_5/sequential_5/dropout_5/Identity?
=model_5/sequential_5/one_hot_encoding_layer_5/PartitionedCallPartitionedCall0model_5/sequential_5/dropout_5/Identity:output:0*
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
__inference_call_260956672?
=model_5/sequential_5/one_hot_encoding_layer_5/PartitionedCall
$model_5/sequential_5/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2&
$model_5/sequential_5/flatten_5/Const
&model_5/sequential_5/flatten_5/ReshapeReshapeFmodel_5/sequential_5/one_hot_encoding_layer_5/PartitionedCall:output:0-model_5/sequential_5/flatten_5/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82(
&model_5/sequential_5/flatten_5/Reshapeĉ
2model_5/sequential_5/dense_5/MatMul/ReadVariableOpReadVariableOp;model_5_sequential_5_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype024
2model_5/sequential_5/dense_5/MatMul/ReadVariableOpô
#model_5/sequential_5/dense_5/MatMulMatMul/model_5/sequential_5/flatten_5/Reshape:output:0:model_5/sequential_5/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#model_5/sequential_5/dense_5/MatMulä
3model_5/sequential_5/dense_5/BiasAdd/ReadVariableOpReadVariableOp<model_5_sequential_5_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype025
3model_5/sequential_5/dense_5/BiasAdd/ReadVariableOpö
$model_5/sequential_5/dense_5/BiasAddBiasAdd-model_5/sequential_5/dense_5/MatMul:product:0;model_5/sequential_5/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2&
$model_5/sequential_5/dense_5/BiasAdd
)model_5/sequential_5/dropout_5/Identity_1Identityinput_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2+
)model_5/sequential_5/dropout_5/Identity_1Ğ
?model_5/sequential_5/one_hot_encoding_layer_5/PartitionedCall_1PartitionedCall2model_5/sequential_5/dropout_5/Identity_1:output:0*
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
__inference_call_260956672A
?model_5/sequential_5/one_hot_encoding_layer_5/PartitionedCall_1Ħ
&model_5/sequential_5/flatten_5/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2(
&model_5/sequential_5/flatten_5/Const_1
(model_5/sequential_5/flatten_5/Reshape_1ReshapeHmodel_5/sequential_5/one_hot_encoding_layer_5/PartitionedCall_1:output:0/model_5/sequential_5/flatten_5/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82*
(model_5/sequential_5/flatten_5/Reshape_1ê
4model_5/sequential_5/dense_5/MatMul_1/ReadVariableOpReadVariableOp;model_5_sequential_5_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype026
4model_5/sequential_5/dense_5/MatMul_1/ReadVariableOpü
%model_5/sequential_5/dense_5/MatMul_1MatMul1model_5/sequential_5/flatten_5/Reshape_1:output:0<model_5/sequential_5/dense_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%model_5/sequential_5/dense_5/MatMul_1è
5model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOpReadVariableOp<model_5_sequential_5_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype027
5model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOpŝ
&model_5/sequential_5/dense_5/BiasAdd_1BiasAdd/model_5/sequential_5/dense_5/MatMul_1:product:0=model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2(
&model_5/sequential_5/dense_5/BiasAdd_1Ħ
(model_5/distance_layer_5/PartitionedCallPartitionedCall-model_5/sequential_5/dense_5/BiasAdd:output:0/model_5/sequential_5/dense_5/BiasAdd_1:output:0*
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
__inference_call_260957412*
(model_5/distance_layer_5/PartitionedCallÛ
IdentityIdentity1model_5/distance_layer_5/PartitionedCall:output:04^model_5/sequential_5/dense_5/BiasAdd/ReadVariableOp6^model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOp3^model_5/sequential_5/dense_5/MatMul/ReadVariableOp5^model_5/sequential_5/dense_5/MatMul_1/ReadVariableOp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2j
3model_5/sequential_5/dense_5/BiasAdd/ReadVariableOp3model_5/sequential_5/dense_5/BiasAdd/ReadVariableOp2n
5model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOp5model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOp2h
2model_5/sequential_5/dense_5/MatMul/ReadVariableOp2model_5/sequential_5/dense_5/MatMul/ReadVariableOp2l
4model_5/sequential_5/dense_5/MatMul_1/ReadVariableOp4model_5/sequential_5/dense_5/MatMul_1/ReadVariableOp:Q M
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
ĉ-
Ħ
__inference_call_26095744	
input
input_1?
;model_5_sequential_5_dense_5_matmul_readvariableop_resource@
<model_5_sequential_5_dense_5_biasadd_readvariableop_resource
identity˘3model_5/sequential_5/dense_5/BiasAdd/ReadVariableOp˘5model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOp˘2model_5/sequential_5/dense_5/MatMul/ReadVariableOp˘4model_5/sequential_5/dense_5/MatMul_1/ReadVariableOp
'model_5/sequential_5/dropout_5/IdentityIdentityinput*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2)
'model_5/sequential_5/dropout_5/Identity?
=model_5/sequential_5/one_hot_encoding_layer_5/PartitionedCallPartitionedCall0model_5/sequential_5/dropout_5/Identity:output:0*
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
__inference_call_260956672?
=model_5/sequential_5/one_hot_encoding_layer_5/PartitionedCall
$model_5/sequential_5/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2&
$model_5/sequential_5/flatten_5/Const
&model_5/sequential_5/flatten_5/ReshapeReshapeFmodel_5/sequential_5/one_hot_encoding_layer_5/PartitionedCall:output:0-model_5/sequential_5/flatten_5/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82(
&model_5/sequential_5/flatten_5/Reshapeĉ
2model_5/sequential_5/dense_5/MatMul/ReadVariableOpReadVariableOp;model_5_sequential_5_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype024
2model_5/sequential_5/dense_5/MatMul/ReadVariableOpô
#model_5/sequential_5/dense_5/MatMulMatMul/model_5/sequential_5/flatten_5/Reshape:output:0:model_5/sequential_5/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#model_5/sequential_5/dense_5/MatMulä
3model_5/sequential_5/dense_5/BiasAdd/ReadVariableOpReadVariableOp<model_5_sequential_5_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype025
3model_5/sequential_5/dense_5/BiasAdd/ReadVariableOpö
$model_5/sequential_5/dense_5/BiasAddBiasAdd-model_5/sequential_5/dense_5/MatMul:product:0;model_5/sequential_5/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2&
$model_5/sequential_5/dense_5/BiasAdd
)model_5/sequential_5/dropout_5/Identity_1Identityinput_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2+
)model_5/sequential_5/dropout_5/Identity_1Ğ
?model_5/sequential_5/one_hot_encoding_layer_5/PartitionedCall_1PartitionedCall2model_5/sequential_5/dropout_5/Identity_1:output:0*
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
__inference_call_260956672A
?model_5/sequential_5/one_hot_encoding_layer_5/PartitionedCall_1Ħ
&model_5/sequential_5/flatten_5/Const_1Const*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2(
&model_5/sequential_5/flatten_5/Const_1
(model_5/sequential_5/flatten_5/Reshape_1ReshapeHmodel_5/sequential_5/one_hot_encoding_layer_5/PartitionedCall_1:output:0/model_5/sequential_5/flatten_5/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82*
(model_5/sequential_5/flatten_5/Reshape_1ê
4model_5/sequential_5/dense_5/MatMul_1/ReadVariableOpReadVariableOp;model_5_sequential_5_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype026
4model_5/sequential_5/dense_5/MatMul_1/ReadVariableOpü
%model_5/sequential_5/dense_5/MatMul_1MatMul1model_5/sequential_5/flatten_5/Reshape_1:output:0<model_5/sequential_5/dense_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%model_5/sequential_5/dense_5/MatMul_1è
5model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOpReadVariableOp<model_5_sequential_5_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype027
5model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOpŝ
&model_5/sequential_5/dense_5/BiasAdd_1BiasAdd/model_5/sequential_5/dense_5/MatMul_1:product:0=model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2(
&model_5/sequential_5/dense_5/BiasAdd_1Ħ
(model_5/distance_layer_5/PartitionedCallPartitionedCall-model_5/sequential_5/dense_5/BiasAdd:output:0/model_5/sequential_5/dense_5/BiasAdd_1:output:0*
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
__inference_call_260957412*
(model_5/distance_layer_5/PartitionedCallÛ
IdentityIdentity1model_5/distance_layer_5/PartitionedCall:output:04^model_5/sequential_5/dense_5/BiasAdd/ReadVariableOp6^model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOp3^model_5/sequential_5/dense_5/MatMul/ReadVariableOp5^model_5/sequential_5/dense_5/MatMul_1/ReadVariableOp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2j
3model_5/sequential_5/dense_5/BiasAdd/ReadVariableOp3model_5/sequential_5/dense_5/BiasAdd/ReadVariableOp2n
5model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOp5model_5/sequential_5/dense_5/BiasAdd_1/ReadVariableOp2h
2model_5/sequential_5/dense_5/MatMul/ReadVariableOp2model_5/sequential_5/dense_5/MatMul/ReadVariableOp2l
4model_5/sequential_5/dense_5/MatMul_1/ReadVariableOp4model_5/sequential_5/dense_5/MatMul_1/ReadVariableOp:O K
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameinput:OK
(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameinput
½
m
V__inference_one_hot_encoding_layer_5_layer_call_and_return_conditional_losses_26095794
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

9
cond_true_26096747
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
ż 
9
__inference_call_26096488
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
cond_false_26096471*
output_shapes	
:*%
then_branchR
cond_true_260964702
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
?

*__inference_model_5_layer_call_fn_26096655
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
E__inference_model_5_layer_call_and_return_conditional_losses_260960522
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
?

/__inference_sequential_5_layer_call_fn_26096699

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
J__inference_sequential_5_layer_call_and_return_conditional_losses_260958702
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
ó

/__inference_sequential_5_layer_call_fn_26095898
input_6
unknown
	unknown_0
identity˘StatefulPartitionedCall˙
StatefulPartitionedCallStatefulPartitionedCallinput_6unknown	unknown_0*
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
J__inference_sequential_5_layer_call_and_return_conditional_losses_260958912
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
_user_specified_name	input_6
?

*__inference_model_5_layer_call_fn_26096573
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
E__inference_model_5_layer_call_and_return_conditional_losses_260961252
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
Ô"
n
N__inference_distance_layer_5_layer_call_and_return_conditional_losses_26095986
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
cond_false_26095969*"
output_shapes
:˙˙˙˙˙˙˙˙˙*%
then_branchR
cond_true_260959682
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


J__inference_sequential_5_layer_call_and_return_conditional_losses_26096676

inputs*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identity˘dense_5/BiasAdd/ReadVariableOp˘dense_5/MatMul/ReadVariableOpw
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout_5/dropout/Const
dropout_5/dropout/MulMulinputs dropout_5/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout_5/dropout/Mulh
dropout_5/dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout_5/dropout/ShapeÓ
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype020
.dropout_5/dropout/random_uniform/RandomUniform
 dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dropout_5/dropout/GreaterEqual/yç
dropout_5/dropout/GreaterEqualGreaterEqual7dropout_5/dropout/random_uniform/RandomUniform:output:0)dropout_5/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
dropout_5/dropout/GreaterEqual
dropout_5/dropout/CastCast"dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout_5/dropout/Cast£
dropout_5/dropout/Mul_1Muldropout_5/dropout/Mul:z:0dropout_5/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout_5/dropout/Mul_1ĉ
(one_hot_encoding_layer_5/PartitionedCallPartitionedCalldropout_5/dropout/Mul_1:z:0*
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
__inference_call_260956672*
(one_hot_encoding_layer_5/PartitionedCalls
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙t  2
flatten_5/Constħ
flatten_5/ReshapeReshape1one_hot_encoding_layer_5/PartitionedCall:output:0flatten_5/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô82
flatten_5/Reshape§
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource* 
_output_shapes
:
ô8*
dtype02
dense_5/MatMul/ReadVariableOp 
dense_5/MatMulMatMulflatten_5/Reshape:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp˘
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_5/BiasAdd?
IdentityIdentitydense_5/BiasAdd:output:0^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
í

2__inference_siamese_model_5_layer_call_fn_26096367
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
M__inference_siamese_model_5_layer_call_and_return_conditional_losses_260961762
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
ż 
9
__inference_call_26096828
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
cond_false_26096811*
output_shapes	
:*%
then_branchR
cond_true_260968102
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

_user_specified_names2"ħL
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
_tf_keras_modelĴ{"class_name": "SiameseModel", "name": "siamese_model_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "SiameseModel"}, "training_config": {"loss": null, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "clipnorm": 1, "learning_rate": 1, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
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
_tf_keras_networkñ{"class_name": "Functional", "name": "model_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1821]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "sequence1"}, "name": "sequence1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1821]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "sequence2"}, "name": "sequence2", "inbound_nodes": []}, {"class_name": "Sequential", "config": {"name": "sequential_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1821]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}, {"class_name": "OneHotEncodingLayer", "config": {"layer was saved without config": true}}, {"class_name": "Flatten", "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 910, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "name": "sequential_5", "inbound_nodes": [[["sequence1", 0, 0, {}]], [["sequence2", 0, 0, {}]]]}, {"class_name": "DistanceLayer", "config": {"layer was saved without config": true}, "name": "distance_layer_5", "inbound_nodes": [[["sequential_5", 1, 0, {"s2": ["sequential_5", 2, 0]}]]]}], "input_layers": [["sequence1", 0, 0], ["sequence2", 0, 0]], "output_layers": {"class_name": "__tuple__", "items": [["distance_layer_5", 0, 0]]}}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1821]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1821]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1821]}, {"class_name": "TensorShape", "items": [null, 1821]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}}
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
_tf_keras_sequential{"class_name": "Sequential", "name": "sequential_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1821]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}, {"class_name": "OneHotEncodingLayer", "config": {"layer was saved without config": true}}, {"class_name": "Flatten", "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 910, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1821]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential"}}
ı
*trainable_variables
+regularization_losses
,	variables
-	keras_api
n__call__
*o&call_and_return_all_conditional_losses
pcall" 
_tf_keras_layer{"class_name": "DistanceLayer", "name": "distance_layer_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
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
ô82dense_5/kernel
:2dense_5/bias
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
_tf_keras_layerĵ{"class_name": "Dropout", "name": "dropout_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}
Ç
7trainable_variables
8regularization_losses
9	variables
:	keras_api
s__call__
*t&call_and_return_all_conditional_losses
ucall"?
_tf_keras_layer{"class_name": "OneHotEncodingLayer", "name": "one_hot_encoding_layer_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
ĉ
;trainable_variables
<regularization_losses
=	variables
>	keras_api
v__call__
*w&call_and_return_all_conditional_losses"×
_tf_keras_layer½{"class_name": "Flatten", "name": "flatten_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
÷

kernel
bias
?trainable_variables
@regularization_losses
A	variables
B	keras_api
x__call__
*y&call_and_return_all_conditional_losses"Ò
_tf_keras_layer¸{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 910, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 7284}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7284]}}
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
ô82Adam/dense_5/kernel/m
 :2Adam/dense_5/bias/m
':%
ô82Adam/dense_5/kernel/v
 :2Adam/dense_5/bias/v
2
2__inference_siamese_model_5_layer_call_fn_26096367
2__inference_siamese_model_5_layer_call_fn_26096285
2__inference_siamese_model_5_layer_call_fn_26096377
2__inference_siamese_model_5_layer_call_fn_26096295²
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
M__inference_siamese_model_5_layer_call_and_return_conditional_losses_26096275
M__inference_siamese_model_5_layer_call_and_return_conditional_losses_26096333
M__inference_siamese_model_5_layer_call_and_return_conditional_losses_26096251
M__inference_siamese_model_5_layer_call_and_return_conditional_losses_26096357²
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
#__inference__wrapped_model_26095751à
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
__inference_call_26096491
__inference_call_26096401Ħ
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
*__inference_model_5_layer_call_fn_26096035
*__inference_model_5_layer_call_fn_26096645
*__inference_model_5_layer_call_fn_26096059
*__inference_model_5_layer_call_fn_26096563
*__inference_model_5_layer_call_fn_26096573
*__inference_model_5_layer_call_fn_26096655À
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
E__inference_model_5_layer_call_and_return_conditional_losses_26096553
E__inference_model_5_layer_call_and_return_conditional_losses_26096635
E__inference_model_5_layer_call_and_return_conditional_losses_26096529
E__inference_model_5_layer_call_and_return_conditional_losses_26096611
E__inference_model_5_layer_call_and_return_conditional_losses_26095996
E__inference_model_5_layer_call_and_return_conditional_losses_26096010À
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
&__inference_signature_wrapper_26096213input_1input_2"
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
/__inference_sequential_5_layer_call_fn_26095877
/__inference_sequential_5_layer_call_fn_26096708
/__inference_sequential_5_layer_call_fn_26096699
/__inference_sequential_5_layer_call_fn_26095898À
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
J__inference_sequential_5_layer_call_and_return_conditional_losses_26095843
J__inference_sequential_5_layer_call_and_return_conditional_losses_26096676
J__inference_sequential_5_layer_call_and_return_conditional_losses_26095855
J__inference_sequential_5_layer_call_and_return_conditional_losses_26096690À
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
3__inference_distance_layer_5_layer_call_fn_26096771¤
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
N__inference_distance_layer_5_layer_call_and_return_conditional_losses_26096765¤
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
__inference_call_26096828
__inference_call_26096885¤
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
,__inference_dropout_5_layer_call_fn_26096912
,__inference_dropout_5_layer_call_fn_26096907´
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
G__inference_dropout_5_layer_call_and_return_conditional_losses_26096902
G__inference_dropout_5_layer_call_and_return_conditional_losses_26096897´
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
;__inference_one_hot_encoding_layer_5_layer_call_fn_26096926
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
V__inference_one_hot_encoding_layer_5_layer_call_and_return_conditional_losses_26096921
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
__inference_call_26096944
__inference_call_26096935
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
,__inference_flatten_5_layer_call_fn_26096955˘
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
G__inference_flatten_5_layer_call_and_return_conditional_losses_26096950˘
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
*__inference_dense_5_layer_call_fn_26096974˘
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
E__inference_dense_5_layer_call_and_return_conditional_losses_26096965˘
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
#__inference__wrapped_model_26095751Z˘W
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
__inference_call_26096401~Z˘W
P˘M
K˘H
"
input/0˙˙˙˙˙˙˙˙˙
"
input/1˙˙˙˙˙˙˙˙˙
Ş "˘

0˙˙˙˙˙˙˙˙˙
__inference_call_26096491fJ˘G
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
__inference_call_26096828K;˘8
1˘.

s1


s2

Ş "	
__inference_call_26096885cK˘H
A˘>

s1˙˙˙˙˙˙˙˙˙

s2˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙i
__inference_call_26096935L+˘(
!˘

x˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙Y
__inference_call_26096944<#˘ 
˘

x

Ş "§
E__inference_dense_5_layer_call_and_return_conditional_losses_26096965^0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙ô8
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 
*__inference_dense_5_layer_call_fn_26096974Q0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙ô8
Ş "˙˙˙˙˙˙˙˙˙Â
N__inference_distance_layer_5_layer_call_and_return_conditional_losses_26096765pK˘H
A˘>

s1˙˙˙˙˙˙˙˙˙

s2˙˙˙˙˙˙˙˙˙
Ş "!˘

0˙˙˙˙˙˙˙˙˙
 
3__inference_distance_layer_5_layer_call_fn_26096771cK˘H
A˘>

s1˙˙˙˙˙˙˙˙˙

s2˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙İ
G__inference_dropout_5_layer_call_and_return_conditional_losses_26096897^4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 İ
G__inference_dropout_5_layer_call_and_return_conditional_losses_26096902^4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 
,__inference_dropout_5_layer_call_fn_26096907Q4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p
Ş "˙˙˙˙˙˙˙˙˙
,__inference_dropout_5_layer_call_fn_26096912Q4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p 
Ş "˙˙˙˙˙˙˙˙˙İ
G__inference_flatten_5_layer_call_and_return_conditional_losses_26096950^4˘1
*˘'
%"
inputs˙˙˙˙˙˙˙˙˙
Ş "&˘#

0˙˙˙˙˙˙˙˙˙ô8
 
,__inference_flatten_5_layer_call_fn_26096955Q4˘1
*˘'
%"
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙ô8à
E__inference_model_5_layer_call_and_return_conditional_losses_26095996f˘c
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
E__inference_model_5_layer_call_and_return_conditional_losses_26096010f˘c
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
E__inference_model_5_layer_call_and_return_conditional_losses_26096529d˘a
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
E__inference_model_5_layer_call_and_return_conditional_losses_26096553d˘a
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
E__inference_model_5_layer_call_and_return_conditional_losses_26096611d˘a
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
E__inference_model_5_layer_call_and_return_conditional_losses_26096635d˘a
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
*__inference_model_5_layer_call_fn_26096035f˘c
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
*__inference_model_5_layer_call_fn_26096059f˘c
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
*__inference_model_5_layer_call_fn_26096563d˘a
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
*__inference_model_5_layer_call_fn_26096573d˘a
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
*__inference_model_5_layer_call_fn_26096645d˘a
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
*__inference_model_5_layer_call_fn_26096655d˘a
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
V__inference_one_hot_encoding_layer_5_layer_call_and_return_conditional_losses_26096921Y+˘(
!˘

x˙˙˙˙˙˙˙˙˙
Ş "*˘'
 
0˙˙˙˙˙˙˙˙˙
 
;__inference_one_hot_encoding_layer_5_layer_call_fn_26096926L+˘(
!˘

x˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙µ
J__inference_sequential_5_layer_call_and_return_conditional_losses_26095843g9˘6
/˘,
"
input_6˙˙˙˙˙˙˙˙˙
p

 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 µ
J__inference_sequential_5_layer_call_and_return_conditional_losses_26095855g9˘6
/˘,
"
input_6˙˙˙˙˙˙˙˙˙
p 

 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 ´
J__inference_sequential_5_layer_call_and_return_conditional_losses_26096676f8˘5
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
J__inference_sequential_5_layer_call_and_return_conditional_losses_26096690f8˘5
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
/__inference_sequential_5_layer_call_fn_26095877Z9˘6
/˘,
"
input_6˙˙˙˙˙˙˙˙˙
p

 
Ş "˙˙˙˙˙˙˙˙˙
/__inference_sequential_5_layer_call_fn_26095898Z9˘6
/˘,
"
input_6˙˙˙˙˙˙˙˙˙
p 

 
Ş "˙˙˙˙˙˙˙˙˙
/__inference_sequential_5_layer_call_fn_26096699Y8˘5
.˘+
!
inputs˙˙˙˙˙˙˙˙˙
p

 
Ş "˙˙˙˙˙˙˙˙˙
/__inference_sequential_5_layer_call_fn_26096708Y8˘5
.˘+
!
inputs˙˙˙˙˙˙˙˙˙
p 

 
Ş "˙˙˙˙˙˙˙˙˙à
M__inference_siamese_model_5_layer_call_and_return_conditional_losses_26096251^˘[
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
M__inference_siamese_model_5_layer_call_and_return_conditional_losses_26096275^˘[
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
M__inference_siamese_model_5_layer_call_and_return_conditional_losses_26096333^˘[
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
M__inference_siamese_model_5_layer_call_and_return_conditional_losses_26096357^˘[
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
2__inference_siamese_model_5_layer_call_fn_26096285^˘[
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
2__inference_siamese_model_5_layer_call_fn_26096295^˘[
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
2__inference_siamese_model_5_layer_call_fn_26096367^˘[
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
2__inference_siamese_model_5_layer_call_fn_26096377^˘[
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
&__inference_signature_wrapper_26096213˘k˘h
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