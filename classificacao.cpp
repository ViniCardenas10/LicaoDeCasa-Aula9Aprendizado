#include "procimagem.h"
#include <vector>

int main()
{
    // 1. Carrega as imagens na resolução original (28x28), invertidas e SEM Bounding Box.
    // Preservar as margens é vital para o HOG calcular as bordas corretamente.
    MNIST mnist(28, true, false);
    mnist.le("/home/vinicius/Documentos/Vinicius/CodigosAula/Inteligentes/MNIST/MNIST_ORG");

    double t1 = tempo();

    // 2. Configuração do Extrator HOG.
    // Estes parâmetros extraem a essência do traço e reduzem os 784 pixels
    // para um vetor matemático super concentrado.
    HOGDescriptor hog;
    hog.winSize = Size(28, 28);
    hog.blockSize = Size(14, 14);
    hog.blockStride = Size(7, 7);
    hog.cellSize = Size(7, 7);
    hog.nbins = 9;

    // Calcula o tamanho exato do vetor HOG que o OpenCV vai cuspir.
    vector<float> desc_temp;
    hog.compute(mnist.AX[0], desc_temp);
    int hog_dims = desc_temp.size();

    // 3. Extrai o HOG de TODAS as 60.000 imagens de treino.
    // Usamos mnist.AX (uchar 0-255) pois o HOG exige esse formato nativo.
    Mat_<float> ax_hog(mnist.na, hog_dims);
    for (int i = 0; i < mnist.na; i++)
    {
        vector<float> desc;
        hog.compute(mnist.AX[i], desc);
        for (int j = 0; j < hog_dims; j++)
        {
            ax_hog(i, j) = desc[j];
        }
    }

    // 4. Extrai o HOG de TODAS as 10.000 imagens de teste.
    Mat_<float> qx_hog(mnist.nq, hog_dims);
    for (int i = 0; i < mnist.nq; i++)
    {
        vector<float> desc;
        hog.compute(mnist.QX[i], desc);
        for (int j = 0; j < hog_dims; j++)
        {
            qx_hog(i, j) = desc[j];
        }
    }

    // 5. Treina a Árvore KD (FlaNN) usando os atributos HOG, não mais os pixels.
    flann::Index ind(ax_hog, flann::KDTreeIndexParams(64));
    double t2 = tempo();

    Mat_<int> matches(mnist.nq, 1);
    Mat_<float> dists(mnist.nq, 1);
    
    // Busca os vizinhos mais próximos no espaço vetorial do HOG.
    ind.knnSearch(qx_hog, matches, dists, 1, flann::SearchParams(256));

    for (int l = 0; l < mnist.qx.rows; l++)
    {
        mnist.qp(l) = mnist.ay(matches(l, 0), 0);
    }
    double t3 = tempo();

    // 6. Imprime os Resultados
    float taxa_erro = 100.0 * mnist.contaErros() / mnist.nq;
    printf("Erros = %10.2f%%\n", taxa_erro);
    printf("Tempo de treinamento: %f segundos\n", t2 - t1);
    printf("Tempo de predicao: %f segundos\n", t3 - t2);

    int n_erros = mnist.contaErros();
    if (n_erros > 0)
    {
        int linhas_img = min(10, (n_erros / 10) + 1);
        int cols_img = min(10, n_erros);

        Mat_<uchar> img_erros = mnist.geraSaidaErros(linhas_img, cols_img);
        imwrite("erros_flann_hog.png", img_erros);
        printf("A imagem 'erros_flann_hog.png' foi salva com os recortes dos erros.\n");
    }

    return 0;
}