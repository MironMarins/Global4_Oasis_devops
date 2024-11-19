using System.ComponentModel.DataAnnotations.Schema;
using System.ComponentModel.DataAnnotations;

namespace ProjetoOasis.Model
{
    [Table("GS_T_CLIENTE")]
    public class Cliente
    {
        [Key]
        [Column("ID")]
        public int Id { get; set; }

        [Required]
        [Column("NOME", TypeName = "VARCHAR2(100)")]
        public string Nome { get; set; }

        [Required]
        [Column("CPF_CNPJ", TypeName = "VARCHAR2(14)")]
        public string CpfCnpj { get; set; }

        [Column("ENDERECO", TypeName = "VARCHAR2(255)")]
        public string Endereco { get; set; }

        [Column("TELEFONE", TypeName = "VARCHAR2(15)")]
        public string Telefone { get; set; }

        [Column("EMAIL", TypeName = "VARCHAR2(100)")]
        public string Email { get; set; }

        [Column("SEGMENTO_CLIENTE")]
        public string SegmentoCliente { get; set; }
    }
}
