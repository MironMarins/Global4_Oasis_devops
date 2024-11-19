using ProjetoOasis.Model;
using Microsoft.EntityFrameworkCore;

namespace ProjetoOasis.Data
{
    public class AppDbContext : DbContext
    {
        public AppDbContext(DbContextOptions<AppDbContext> options) : base(options) { }

        public DbSet<Produto> Produtos { get; set; }
        public DbSet<Cliente> Clientes { get; set; }
    }
}
