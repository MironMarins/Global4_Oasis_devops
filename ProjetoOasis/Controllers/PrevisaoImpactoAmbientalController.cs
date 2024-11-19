using Microsoft.AspNetCore.Mvc;
using Microsoft.ML.Data;
using Microsoft.ML;
using System.IO;
using System;

namespace ProjetoOasis.Controllers
{
    // Define a classe de dados de entrada
    public class DadosProduto
    {
        [LoadColumn(0)] public string NomeProduto { get; set; }
        [LoadColumn(1)] public string Categoria { get; set; }
        [LoadColumn(2)] public float Preco { get; set; }
        [LoadColumn(3)] public float QuantidadeVendida { get; set; } // Nova variável
        [LoadColumn(4)] public string Origem { get; set; } // Nova variável
        [LoadColumn(5)] public float Litros { get; set; } // Nova variável

        // Coluna de impacto ambiental, que será prevista
        [LoadColumn(6)] public float ImpactoAmbiental { get; set; }  // Nova variável para impacto ambiental
    }

    // Define a classe de saída de previsão
    public class PrevisaoImpactoAmbiental
    {
        [ColumnName("PredictedLabel")]
        public float ImpactoAmbiental { get; set; } // Valor previsto para o impacto ambiental
    }

    [Route("api/[controller]")]
    [ApiController]
    public class PrevisaoImpactoAmbientalController : ControllerBase
    {
        private readonly string caminhoModelo = Path.Combine(Environment.CurrentDirectory, "wwwroot", "MLModels", "ModeloImpactoAmbiental.zip");
        private readonly string caminhoTreinamento = Path.Combine(Environment.CurrentDirectory, "Data", "dados_treinamento.csv");
        private readonly MLContext mlContext;

        public PrevisaoImpactoAmbientalController()
        {
            mlContext = new MLContext();

            if (!System.IO.File.Exists(caminhoModelo))
            {
                Console.WriteLine("Modelo não encontrado. Iniciando treinamento");
                TreinarModelo();
            }
        }

        // Método para treinar o modelo
        private void TreinarModelo()
        {
            var pastaModelo = Path.GetDirectoryName(caminhoModelo);
            if (!Directory.Exists(pastaModelo))
            {
                Directory.CreateDirectory(pastaModelo);
                Console.WriteLine($"Diretório criado: {pastaModelo}");
            }

            // Carregando os dados de treinamento
            IDataView dadosTreinamento = mlContext.Data.LoadFromTextFile<DadosProduto>(
                path: caminhoTreinamento, hasHeader: true, separatorChar: ',');

            // Pipeline ajustado para regressão
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", nameof(DadosProduto.ImpactoAmbiental))  // Rótulo ajustado
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("CategoriaEncoded", nameof(DadosProduto.Categoria)))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("OrigemEncoded", nameof(DadosProduto.Origem)))
                .Append(mlContext.Transforms.Concatenate("Features", "CategoriaEncoded", "OrigemEncoded", nameof(DadosProduto.Preco), nameof(DadosProduto.QuantidadeVendida), nameof(DadosProduto.Litros)))
                .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Label", featureColumnName: "Features"))  // Regressão para prever impacto ambiental
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));

            // Treinando o modelo
            var modelo = pipeline.Fit(dadosTreinamento);

            // Salvando o modelo treinado
            mlContext.Model.Save(modelo, dadosTreinamento.Schema, caminhoModelo);
            Console.WriteLine($"Modelo treinado e salvo em: {caminhoModelo}");
        }

        // Método para prever o impacto ambiental de um produto
        [HttpPost("prever")]
        public ActionResult<PrevisaoImpactoAmbiental> PreverImpactoAmbiental([FromBody] DadosProduto dados)
        {
            if (!System.IO.File.Exists(caminhoModelo))
            {
                return BadRequest("O modelo ainda não foi treinado.");
            }

            ITransformer modelo;
            try
            {
                using (var stream = new FileStream(caminhoModelo, FileMode.Open, FileAccess.Read, FileShare.Read))
                {
                    modelo = mlContext.Model.Load(stream, out var modeloSchema);
                }
            }
            catch (Exception ex)
            {
                return StatusCode(500, $"Erro ao carregar o modelo: {ex.Message}");
            }

            var enginePrevisao = mlContext.Model.CreatePredictionEngine<DadosProduto, PrevisaoImpactoAmbiental>(modelo);

            PrevisaoImpactoAmbiental previsao;
            try
            {
                previsao = enginePrevisao.Predict(dados);
            }
            catch (Exception ex)
            {
                return StatusCode(500, $"Erro ao prever o impacto ambiental: {ex.Message}");
            }

            if (previsao == null)
            {
                return Ok(new PrevisaoImpactoAmbiental { ImpactoAmbiental = -1 }); // Valor default indicando erro
            }

            return Ok(previsao);
        }
    }
}
