using System.ComponentModel.DataAnnotations.Schema;
using System.ComponentModel.DataAnnotations;

namespace ProjetoOasis.Model
{
    [Table("GS_T_PRODUTO")]
    public class Produto
    {
        [Key]
        [Column("ID")]
        public int Id { get; set; }

        [Required]
        [Column("NOME", TypeName = "VARCHAR2(100)")]
        public string Nome { get; set; }

        [Required]
        [Column("PRECO", TypeName = "NUMBER(10,2)")]
        public decimal Preco { get; set; }

        [Required]
        [Column("CATEGORIA", TypeName = "VARCHAR2(50)")]
        public string Categoria { get; set; }

        [Column("QUANTIDADE_VENDIDA")]
        public int QuantidadeVendida { get; set; }

        [Column("LITROS", TypeName = "NUMBER(10,2)")]
        public decimal? Litros { get; set; }

        [Column("IMPACTO_AMBIENTAL")]
        public decimal ImpactoAmbiental { get; set; }  // Impacto de 0 a 10, sendo 0 o menor impacto

        [Column("ORIGEM")]
        public string Origem { get; set; }  // Nacional ou Importado
    }
}