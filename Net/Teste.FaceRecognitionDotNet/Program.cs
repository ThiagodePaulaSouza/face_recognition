using System.Data;
using System.Data.SqlClient;
using Dapper;
using FaceRecognitionDotNet;

var directory = @"E:\PROJETOS\face_recognition\Net\Teste.FaceRecognitionDotNet\Models\dlib_face_recognition_resnet_model_v1.dat";


var image = @"E:\PROJETOS\face_recognition\Net\Teste.FaceRecognitionDotNet\Images\teste1.jpg";

var repo = new FaceRepository("");

using var fr = FaceRecognition.Create(directory);

List<(string image, List<FaceMatchResult>)> result = new();

using var im = FaceRecognition.LoadImageFile(image);
var locations = fr.FaceLocations(im);
foreach (var l in locations)
{
    FaceEncoding? encodings = fr.FaceEncodings(im, new[] { l }).FirstOrDefault();
    if (encodings == null) continue;

    var faceEncodingArray = encodings.GetRawEncoding().Select(d => (float)d);
    List<FaceMatchResult> res = repo.SearchTopMatchesAsync((List<float[]>)faceEncodingArray).Result;

    result.Add((image, res));
}

public class FaceRepository
{
    private readonly string _connectionString;

    public FaceRepository(string connectionString)
    {
        _connectionString = connectionString;
    }

    public async Task<List<FaceMatchResult>> SearchTopMatchesAsync(
        List<float[]> embeddings,
        int top = 5)
    {
        using var connection = new SqlConnection(_connectionString);

        // monta JSON com os embeddings
        var inputs = embeddings
            .Select((e, i) => new
            {
                Id = i,
                Vector = e
            });

        var json = System.Text.Json.JsonSerializer.Serialize(inputs);

        var sql = @"
DECLARE @inputs NVARCHAR(MAX) = @json;

-- converte JSON em tabela
WITH InputVectors AS (
    SELECT 
        JSON_VALUE(value, '$.Id') AS Id,
        CAST(JSON_QUERY(value, '$.Vector') AS VECTOR(512)) AS Encoding
    FROM OPENJSON(@inputs)
)

SELECT 
    i.Id AS InputId,
    f.Id AS FaceId,
    f.PessoaId,
    VECTOR_DISTANCE('cosine', f.Encoding, i.Encoding) AS Distance
FROM InputVectors i
CROSS APPLY (
    SELECT TOP (@top) *
    FROM Face f
    ORDER BY VECTOR_DISTANCE('cosine', f.Encoding, i.Encoding)
) f
ORDER BY i.Id, Distance;
";

        var result = await connection.QueryAsync<FaceMatchResult>(
            sql,
            new
            {
                json,
                top
            });

        return result.ToList();
    }
}
public class FaceEmbedding
{
    public Guid Id { get; set; }
    public Guid PessoaId { get; set; }
    public float[] Encoding { get; set; }
}

public class FaceMatchResult
{
    public int InputId { get; set; }
    public Guid FaceId { get; set; }
    public Guid PessoaId { get; set; }
    public double Distance { get; set; }
}