#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>


cv::CascadeClassifier revolverIdentifier;

int main(int argc, const char **argv)
{
	
    cv::CommandLineParser parser(argc, argv,
        "{image_path|./test.jpg|}"
        "{camera|0|Camera device number.}");
    parser.about("\nDemostracion de clasificador entrenado para detectar revolveres\n\n");
    parser.printMessage();

    //Cargamos el clasificador
    if (!revolverIdentifier.load("cascade.xml"))
    {
        std::cout << "Error cargando el clasificador\n";
        return -1;
    }

    //Leemos la imagen que se paso por linea de comandos, si no se paso ninguna se toma test.jpg
    cv::Mat image = cv::imread(parser.get<std::string>("image_path"));

    //Convertimos a blanco y negro y ecualizamos el histograma
    cv::Mat imageGrayScale;
    cvtColor(image, imageGrayScale, cv::COLOR_BGR2GRAY);
    equalizeHist(imageGrayScale, imageGrayScale);

    //vector para las cajas de los revolveres en la imagen
    std::vector<cv::Rect> revolveres;

    //detectamos los revolveres
    revolverIdentifier.detectMultiScale(imageGrayScale, revolveres);

    //Dibujamos los rectangulos que marcan los revolveres
    for (size_t i = 0; i < revolveres.size(); i++)
    {
        cv::rectangle(image, revolveres[i], cv::Scalar(0, 0, 255), 4);
    }

    //Guardamos la imagen y la mostramos
    cv::imwrite("./output.png", image);

    cv::imshow("Output", image);

    cv::waitKey();

    return 0;


}
