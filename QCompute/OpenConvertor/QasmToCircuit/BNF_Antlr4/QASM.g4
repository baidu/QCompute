grammar QASM;

mainprog
  : version include statement*
  ;

statement 
  : decl 
  | gatedecl goplist '}'
  | gatedecl '}'
  | 'opaque' ID idlist ';'
  | 'opaque' ID '(' ')' idlist ';'
  | 'opaque' ID '(' idlist ')' idlist ';'
  | qop
  | comments
  ;

version
  : 'OPENQASM' REAL ';'
  ;

include
  : 'include' '"' FILE_NAME '"' ';'
  ;

decl
  : 'qreg' ID '[' INT ']' ';'       # qReg
  | 'creg' ID '[' INT ']' ';'       # cReg
  ;

gatedecl
  : 'gate' ID idlist '{'
  | 'gate' ID '(' ')' idlist '{'
  | 'gate' ID '(' idlist ')' idlist '{'
  ;

goplist 
  : (uop | barrierOp | comments)+
  ;

qop
  : uop
  | measureOp
  | barrierOp
  ;

uop
  : ID anylist ';'
  | ID '(' ')' anylist ';'
  | ID '(' explist ')' anylist ';'
  ;

comments
  : lineComment
  | multipleComments
  ;

lineComment
  : LINE_COMMENT
  ;

multipleComments
  : COMMENT
  ;

measureOp
  : 'measure' argument '->' argument ';'
  ;

barrierOp
  : 'barrier' anylist ';'
  ;

anylist
  : idlist 
  | mixedlist
  ;

idlist 
  : (ID ',')* ID
  ;

mixedlist
  : (ID '[' INT ']' ',')* ID
  | (ID '[' INT ']' ',')* ID '[' INT ']'
  | ((ID ',')* ID ',')* ID '[' INT ']'
  ;

argument 
  : ID 
  | ID '[' INT ']' 
  ;

explist 
  : (exp ',')* exp 
  ;

exp
  : REAL
  | INT 
  | 'pi'
  | ID
  | exp '+' exp
  | exp '-' exp 
  | exp '*' exp
  | exp '/' exp
  | '-' exp 
  | exp '^' exp
  | '(' exp ')'
  | unaryop '(' exp ')' 
  ;

unaryop 
  : sin | cos | tan | expx | ln | sqrt
  ;

sin
  : 'sin'
  ;

cos
  : 'cos'
  ;

tan
  : 'tan'
  ;

expx
  : 'exp'
  ;

ln
  : 'ln'
  ;

sqrt
  : 'sqrt'
  ;

// Other / REGEX Tokens

ID
  : [A-Za-z_][A-Za-z0-9_]*
  ;

REAL
  : INT '.' [0-9]+ ([eE][+-]? [0-9]+)?
  ;

INT
  : [+-]? [0-9]+
  ;

WS
  : [ \t\u000C\r\n]+ -> skip
  ;

COMMENT
  : '/*' .*? '*/'
  ;

LINE_COMMENT
  : '//' ~[\r\n]*
  ;

FILE_NAME
  : [A-Za-z0-9_.]+
  ;