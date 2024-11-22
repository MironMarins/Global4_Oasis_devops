--------------------------------------------------------
--  Arquivo criado - Quinta-feira-Novembro-21-2024   
--------------------------------------------------------
--------------------------------------------------------
--  DDL for Table GS_T_PRODUTO
--------------------------------------------------------

  CREATE TABLE "RM98959"."GS_T_PRODUTO" 
   (	"ID" NUMBER(10,0), 
	"NOME" VARCHAR2(100 BYTE), 
	"PRECO" NUMBER(10,2), 
	"CATEGORIA" VARCHAR2(50 BYTE), 
	"QUANTIDADE_VENDIDA" NUMBER(10,0) DEFAULT 0, 
	"LITROS" NUMBER(10,2), 
	"IMPACTO_AMBIENTAL" NUMBER(5,2), 
	"ORIGEM" VARCHAR2(50 BYTE)
   ) SEGMENT CREATION IMMEDIATE 
  PCTFREE 10 PCTUSED 40 INITRANS 1 MAXTRANS 255 
 NOCOMPRESS LOGGING
  STORAGE(INITIAL 65536 NEXT 1048576 MINEXTENTS 1 MAXEXTENTS 2147483645
  PCTINCREASE 0 FREELISTS 1 FREELIST GROUPS 1
  BUFFER_POOL DEFAULT FLASH_CACHE DEFAULT CELL_FLASH_CACHE DEFAULT)
  TABLESPACE "TBSPC_ALUNOS" ;
--------------------------------------------------------
--  DDL for Table GS_T_CLIENTE
--------------------------------------------------------

  CREATE TABLE "RM98959"."GS_T_CLIENTE" 
   (	"ID" NUMBER(10,0), 
	"NOME" VARCHAR2(100 BYTE), 
	"CPF_CNPJ" VARCHAR2(14 BYTE), 
	"ENDERECO" VARCHAR2(255 BYTE), 
	"TELEFONE" VARCHAR2(15 BYTE), 
	"EMAIL" VARCHAR2(100 BYTE), 
	"SEGMENTO_CLIENTE" VARCHAR2(50 BYTE)
   ) SEGMENT CREATION IMMEDIATE 
  PCTFREE 10 PCTUSED 40 INITRANS 1 MAXTRANS 255 
 NOCOMPRESS LOGGING
  STORAGE(INITIAL 65536 NEXT 1048576 MINEXTENTS 1 MAXEXTENTS 2147483645
  PCTINCREASE 0 FREELISTS 1 FREELIST GROUPS 1
  BUFFER_POOL DEFAULT FLASH_CACHE DEFAULT CELL_FLASH_CACHE DEFAULT)
  TABLESPACE "TBSPC_ALUNOS" ;
--------------------------------------------------------
--  DDL for Index SYS_C004099597
--------------------------------------------------------

  CREATE UNIQUE INDEX "RM98959"."SYS_C004099597" ON "RM98959"."GS_T_PRODUTO" ("ID") 
  PCTFREE 10 INITRANS 2 MAXTRANS 255 COMPUTE STATISTICS 
  STORAGE(INITIAL 65536 NEXT 1048576 MINEXTENTS 1 MAXEXTENTS 2147483645
  PCTINCREASE 0 FREELISTS 1 FREELIST GROUPS 1
  BUFFER_POOL DEFAULT FLASH_CACHE DEFAULT CELL_FLASH_CACHE DEFAULT)
  TABLESPACE "TBSPC_ALUNOS" ;
--------------------------------------------------------
--  DDL for Index SYS_C004099587
--------------------------------------------------------

  CREATE UNIQUE INDEX "RM98959"."SYS_C004099587" ON "RM98959"."GS_T_CLIENTE" ("ID") 
  PCTFREE 10 INITRANS 2 MAXTRANS 255 COMPUTE STATISTICS 
  STORAGE(INITIAL 65536 NEXT 1048576 MINEXTENTS 1 MAXEXTENTS 2147483645
  PCTINCREASE 0 FREELISTS 1 FREELIST GROUPS 1
  BUFFER_POOL DEFAULT FLASH_CACHE DEFAULT CELL_FLASH_CACHE DEFAULT)
  TABLESPACE "TBSPC_ALUNOS" ;
--------------------------------------------------------
--  Constraints for Table GS_T_PRODUTO
--------------------------------------------------------

  ALTER TABLE "RM98959"."GS_T_PRODUTO" MODIFY ("NOME" NOT NULL ENABLE);
  ALTER TABLE "RM98959"."GS_T_PRODUTO" MODIFY ("PRECO" NOT NULL ENABLE);
  ALTER TABLE "RM98959"."GS_T_PRODUTO" MODIFY ("CATEGORIA" NOT NULL ENABLE);
  ALTER TABLE "RM98959"."GS_T_PRODUTO" MODIFY ("IMPACTO_AMBIENTAL" NOT NULL ENABLE);
  ALTER TABLE "RM98959"."GS_T_PRODUTO" ADD PRIMARY KEY ("ID")
  USING INDEX PCTFREE 10 INITRANS 2 MAXTRANS 255 COMPUTE STATISTICS 
  STORAGE(INITIAL 65536 NEXT 1048576 MINEXTENTS 1 MAXEXTENTS 2147483645
  PCTINCREASE 0 FREELISTS 1 FREELIST GROUPS 1
  BUFFER_POOL DEFAULT FLASH_CACHE DEFAULT CELL_FLASH_CACHE DEFAULT)
  TABLESPACE "TBSPC_ALUNOS"  ENABLE;
--------------------------------------------------------
--  Constraints for Table GS_T_CLIENTE
--------------------------------------------------------

  ALTER TABLE "RM98959"."GS_T_CLIENTE" MODIFY ("NOME" NOT NULL ENABLE);
  ALTER TABLE "RM98959"."GS_T_CLIENTE" MODIFY ("CPF_CNPJ" NOT NULL ENABLE);
  ALTER TABLE "RM98959"."GS_T_CLIENTE" ADD PRIMARY KEY ("ID")
  USING INDEX PCTFREE 10 INITRANS 2 MAXTRANS 255 COMPUTE STATISTICS 
  STORAGE(INITIAL 65536 NEXT 1048576 MINEXTENTS 1 MAXEXTENTS 2147483645
  PCTINCREASE 0 FREELISTS 1 FREELIST GROUPS 1
  BUFFER_POOL DEFAULT FLASH_CACHE DEFAULT CELL_FLASH_CACHE DEFAULT)
  TABLESPACE "TBSPC_ALUNOS"  ENABLE;

--------------------------------------------------------
--  DDL for Table GS_T_VENDA
--------------------------------------------------------

CREATE TABLE "RM98959"."GS_T_VENDA" (
    "ID" NUMBER(10, 0) PRIMARY KEY,
    "ID_PRODUTO" NUMBER(10, 0) NOT NULL,
    "ID_CLIENTE" NUMBER(10, 0) NOT NULL,
    "QUANTIDADE" NUMBER(10, 0),
    "DATA_VENDA" DATE,
    -- Other relevant fields like total price, discount, etc.
    
    CONSTRAINT "FK_VENDA_PRODUTO" FOREIGN KEY ("ID_PRODUTO")
        REFERENCES "RM98959"."GS_T_PRODUTO" ("ID"),
    CONSTRAINT "FK_VENDA_CLIENTE" FOREIGN KEY ("ID_CLIENTE")
        REFERENCES "RM98959"."GS_T_CLIENTE" ("ID")
);
