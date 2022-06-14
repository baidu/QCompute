grammar QASMOriginal;

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
  | 'if' '(' ID '==' INT ')' qop
  ;

version
  : 'OPENQASM' REAL ';'
  ;

include
  : 'include' '"' FILE_NAME '"' ';'
  ;

decl
  : 'qreg' ID '[' INT ']' ';'
  | 'creg' ID '[' INT ']' ';'
  ;

gatedecl
  : 'gate' ID idlist '{'
  | 'gate' ID '(' ')' idlist '{'
  | 'gate' ID '(' idlist ')' idlist '{'
  ;

goplist 
  : (uop | barrierOp)+
  ;

qop
  : uop
  | measureOp
  | barrierOp
  | resetOp
  ;

uop
  : 'U' '(' explist ')' argument ';'
  | 'CX' argument ',' argument ';'
  | ID anylist ';'
  | ID '(' ')' anylist ';'
  | ID '(' explist ')' anylist ';'
  ;

measureOp
  : 'measure' argument '->' argument ';'
  ;

barrierOp
  : 'barrier' anylist ';'
  ;

resetOp
  : 'reset' argument ';'
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
  : 'sin' | 'cos' | 'tan' | 'exp' | 'ln' | 'sqrt'
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
  : '/*' .*? '*/' -> skip
  ;

LINE_COMMENT
  : '//' ~[\r\n]* -> skip
  ;

FILE_NAME
  : [A-Za-z0-9_.]+
  ;