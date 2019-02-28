#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <vector>
#include <unistd.h>
#include <cmath>
#include <iomanip>
// Ubuntuなら
// #include <sys/stat.h>
// Windowsなら
// #include <direct.h>

#include <GL/glew.h>
#include <SDL2/SDL.h>
#include <glm/glm.hpp>
#include <glm/ext.hpp>

// Read file
#include <iostream>
#include <fstream>
#include <string>
#include "picojson.h" //同じフォルダ内のヘッダファイルの読み込み

#include <opencv2/opencv.hpp>
#include <opencv/highgui.h>
using namespace std;

////////////////////////////////////////////////////////////////////////////////
// キャスト変換 (点群読み込み用)
float stofTryCatch(const std::string& a, bool* is_float) {
  float result = 0.0f;
  try {
    result = std::stof(a);
  } catch (const std::invalid_argument&) {
    // std::cout << "Error: The string '" << a << "' is not float." << std::endl;
    *is_float = false;
    return result;
  } catch (const std::out_of_range&) {
    // std::cout << "Error: The string '" << a << "' is float but out of range." << std::endl;
    *is_float = false;
    return result;
  }
  *is_float = true;
  // std::cout << "'" << a << "' -> " << result << std::endl;
  return result;
}
////////////////////////////////////////////////////////////////////////////////
//Split関数 (点群読み込み用)
vector<string> split(const string& src, const char* delim = " ") { //スペース区切り
    vector<string> vec;
    string::size_type len = src.length();

    for (string::size_type i = 0, n; i < len; i = n + 1) {
        n = src.find_first_of(delim, i);
        if (n == string::npos) {
            n = len;
        }
        vec.push_back(src.substr(i, n - i));
    }

    return vec;
}
////////////////////////////////////////////////////////////////////////////////

void GLAPIENTRY mygldbgcallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam) {
	printf("[GLDBG]: %s\n", message);
	if (type == GL_DEBUG_TYPE_ERROR) {
		abort();
	}
}
////////////////////////////////////////////////////////////////////////////////
void saveImage( const unsigned int img_w, const unsigned int img_h, int frame_num, std::string dir, std::string dataset_name, int degree) {
  const unsigned int ch = 3;
  void* dataBuffer = NULL;
  dataBuffer = (GLubyte*)malloc( img_w * img_h *ch);

  glReadBuffer(GL_BACK);
  glReadPixels(
    0, // lower left x is ?
    0, // lower left y is ?
    img_w,
    img_h,
    GL_BGR, // Read color
    GL_UNSIGNED_BYTE, // Read data type
    dataBuffer //bit map pixels pointer
  );

  GLubyte* p = static_cast<GLubyte*>(dataBuffer);
  IplImage* outImage = cvCreateImage( cvSize(img_w, img_h), IPL_DEPTH_8U, 3);

  for ( unsigned int j = 0; j < img_h; ++j)
  {
    for ( unsigned int i = 0; i < img_w; ++i)
    {
      outImage->imageData[ (img_h - j - 1 ) * outImage->widthStep + i * 3 + 0 ] = *p;
      outImage->imageData[ (img_h - j - 1 ) * outImage->widthStep + i * 3 + 1 ] = *( p + 1 );
      outImage->imageData[ (img_h - j - 1 ) * outImage->widthStep + i * 3 + 2 ] = *( p + 2 );
      p += 3;
    }
  }
  // Ubuntu
  // if (_mkdir("./images/")==0) {
  //   std::cout << "make image folder" << std::endl;
  // }
  //123の頭に5個0を詰めて8桁にする
  std::ostringstream name;
  std::ostringstream pc_img_dir;
  std::ostringstream ss;

	name << std::setw(10) << std::setfill('0') << frame_num;

  if (degree == 0) {
    // pc_img_dir << dir << "pc_images/";
    pc_img_dir << dir << "pc_images/";
    ss  << pc_img_dir.str() << dataset_name << "_" << name.str() << ".png";
  } else {
    pc_img_dir << dir << "pc_images_" << degree << "/";
    ss  << pc_img_dir.str() << dataset_name << "_" << degree << "_" << name.str() << ".png";
  }

  std::string filename = ss.str();
  std::cout << filename << std::endl;
  cvSaveImage( filename.c_str(), outImage, 0);

}
//////////////////////////////////////////////////////////////////////////////


#undef main
int main() {
  unsigned int width = 1392;
  unsigned int height = 512;


	//SDL初期設定
	SDL_Init(SDL_INIT_VIDEO);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_COMPATIBILITY); //Compatibiltyプロファイル指定にしてClassicな機能を使う
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_DEBUG_FLAG);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

  // SDL_CreateWindow(const char* title, int screen_x, int screen_y, int w, int h, Uint32 flags)
  SDL_Window *window = SDL_CreateWindow("FixedFunctionShaderTest", width/2, height/2, width, height, SDL_WINDOW_SHOWN | SDL_WINDOW_OPENGL);
  SDL_GLContext context = SDL_GL_CreateContext(window);

	//glew初期化＠GLコンテキスト作った後
	glewInit();

	//OpenGLでエラーが起きた場合のコールバック
	glDebugMessageCallback(mygldbgcallback, NULL);

	//描画領域の大きさ
	int dwindowWidth, dwindowHeight;
	SDL_GL_GetDrawableSize(window, &dwindowWidth, &dwindowHeight);

  std::string dataset_name = "kitti_20110930_0016";
  float degree;
  std::string dir_name = "./" + dataset_name + "/";

	///////////////////// 点群を読み込む ////////////////////////////////////////////
	std::fstream ifspc(dir_name + "depthmaps/merged_down.ply"); //点群ファイル読み込み

	std::string str;
	std::vector<float> pc_position_buf;
	std::vector<float> pc_color_buf;
	int count = 0;
	bool is_float;

	if(ifspc.fail()) {
		std::cout << "read ply Failed" << std::endl;
		return -1;
	} else {
    std::cout << "Reading merge.ply" << std::endl;
  }

  // 点群読み込み Stringで読み込んでから、floatに変換
	while( (getline(ifspc, str)) ) {
		count++;
		// if (count  > 14) { //点群読み込み開始位置
    if (count  > 16) { //点群読み込み開始位置
			vector<string> vec = split(str);

			float x = stofTryCatch(vec[0], &is_float);
			float y = stofTryCatch(vec[1], &is_float);
			float z = stofTryCatch(vec[2], &is_float);
			// float nx = stofTryCatch(vec[3], &is_float);
			// float ny = stofTryCatch(vec[4], &is_float);
			// float nz = stofTryCatch(vec[5], &is_float);
			// float r = stofTryCatch(vec[6], &is_float) / 255;
			// float g = stofTryCatch(vec[7], &is_float) / 255;
			// float b = stofTryCatch(vec[8], &is_float) / 255;
      //
      //cloud compareのフォーマット
      float r = stofTryCatch(vec[3], &is_float) / 255;
			float g = stofTryCatch(vec[4], &is_float) / 255;
			float b = stofTryCatch(vec[5], &is_float) / 255;

			pc_position_buf.push_back(x); //x
			pc_position_buf.push_back(y); //z
			pc_position_buf.push_back(z); //y
			//色は0~1で指定
			pc_color_buf.push_back(r); //r
			pc_color_buf.push_back(g); //g
			pc_color_buf.push_back(b); //b
		}
	}
  std::cout << "Finish reading ply" << std::endl;

  ///////////////////// カメラ軌道確認用 四角錐 //////////////////////////////////
  std::vector<float> cube0_pos_buf;
  std::vector<float> cube0_col_buf;

  {
    float tvbuf[]={
      0,0,0, //頂点
      -100,100,100,
      0,0,0,
      -100,-100,100,
      0,0,0,
      100,-100,100,
      0,0,0,
      100,100,100,
      -100,100,100,
      -100,-100,100,
      -100,-100,100,
      100,-100,100,
      100,-100,100,
      100,100,100,
      100,100,100,
      -100,100,100
    };
    cube0_pos_buf.clear(); cube0_pos_buf.resize(sizeof(tvbuf)/sizeof(float));
    memcpy(&cube0_pos_buf[0],tvbuf,sizeof(tvbuf));
  }

  // color
  {
    float tcbuf[]={
      1,1,1,
      1,1,1,
      1,1,1,
      1,1,1,
      1,1,1,
      1,1,1,
      1,1,1,
      1,1,1,
      1,1,1,
      1,1,1,
      1,1,1,
      1,1,1,
      1,1,1,
      1,1,1,
      1,1,1,
      1,1,1
    };
    cube0_col_buf.clear(); cube0_col_buf.resize(sizeof(tcbuf)/sizeof(float));
    memcpy(&cube0_col_buf[0],tcbuf,sizeof(tcbuf));
  }

  // VAO
  GLuint cam_vao;
	glGenVertexArrays(1, &cam_vao);
	glBindVertexArray(cam_vao);
	GLuint cam_vbos[2];
	glGenBuffers(2, cam_vbos);
	glBindBuffer(GL_ARRAY_BUFFER, cam_vbos[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * cube0_pos_buf.size(), &cube0_pos_buf[0], GL_STATIC_DRAW); //データ転送
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, sizeof(float) * 3, (GLvoid*)0);
	glBindBuffer(GL_ARRAY_BUFFER, cam_vbos[1]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * cube0_col_buf.size(), &cube0_col_buf[0], GL_STATIC_DRAW); //データ転送
	glEnableClientState(GL_COLOR_ARRAY);
	glColorPointer(3, GL_FLOAT, sizeof(float) * 3, (GLvoid*)0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, pc_vbos[2]); //インデックス配列付ける場合
	//glBufferData(GL_ARRAY_BUFFER, sizeof(float) * pc_indices_buf.size(), &pc_indices_buf[0], GL_STATIC_DRAW); //データ転送
	glBindVertexArray(0);
	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);

  ///////////////////// 床  //////////////////////////////////
  std::vector<float> ground_pos_buf;
  std::vector<float> ground_col_buf;

  {
    float g_buf[]={
      //
      -1000,-5000,-20, //頂点
      10000,-5000,-20,
      //
      -1000,5000,-20,
      10000,5000,-20,

    };
    ground_pos_buf.clear(); ground_pos_buf.resize(sizeof(g_buf)/sizeof(float));
    memcpy(&ground_pos_buf[0],g_buf,sizeof(g_buf));
  }

  // color
  {
    float g_c_buf[]={
      //
      0.4, 0.4, 0.4,
      0.4, 0.4, 0.4,
      0.4, 0.4, 0.4,
      0.4, 0.4, 0.4,
    };
    ground_col_buf.clear(); ground_col_buf.resize(sizeof(g_c_buf)/sizeof(float));
    memcpy(&ground_col_buf[0],g_c_buf,sizeof(g_c_buf));
  }

  // VAO
  GLuint ground_vao;
	glGenVertexArrays(1, &ground_vao);
	glBindVertexArray(ground_vao);
	GLuint ground_vbos[2];
	glGenBuffers(2, ground_vbos);
	glBindBuffer(GL_ARRAY_BUFFER, ground_vbos[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * ground_pos_buf.size(), &ground_pos_buf[0], GL_STATIC_DRAW); //データ転送
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, sizeof(float) * 3, (GLvoid*)0);
	glBindBuffer(GL_ARRAY_BUFFER, ground_vbos[1]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * ground_col_buf.size(), &ground_col_buf[0], GL_STATIC_DRAW); //データ転送
	glEnableClientState(GL_COLOR_ARRAY);
	glColorPointer(3, GL_FLOAT, sizeof(float) * 3, (GLvoid*)0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, pc_vbos[2]); //インデックス配列付ける場合
	//glBufferData(GL_ARRAY_BUFFER, sizeof(float) * pc_indices_buf.size(), &pc_indices_buf[0], GL_STATIC_DRAW); //データ転送
	glBindVertexArray(0);
	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);

  ///////////////////// 空 //////////////////////////////////
  std::vector<float> sky_pos_buf;
  std::vector<float> sky_col_buf;

  {
    float sky_buf[]={
      //
      -50,-5000,100, //頂点
      600,-5000,100,
      //
      -500,5000,100,
      600,5000,100,

    };
    sky_pos_buf.clear(); sky_pos_buf.resize(sizeof(sky_buf)/sizeof(float));
    memcpy(&sky_pos_buf[0],sky_buf,sizeof(sky_buf));
  }

  // color
  {
    float sky_c_buf[]={
      //
      // 0.0, 0.8, 1,
      // 0.0, 0.8, 1,
      // 0.0, 0.8, 1,
      // 0.0, 0.8, 1,
      1,1,1,
      1,1,1,
      1,1,1,
      1,1,1,

    };
    sky_col_buf.clear(); sky_col_buf.resize(sizeof(sky_c_buf)/sizeof(float));
    memcpy(&sky_col_buf[0],sky_c_buf,sizeof(sky_c_buf));
  }

  // VAO
  GLuint sky_vao;
	glGenVertexArrays(1, &sky_vao);
	glBindVertexArray(sky_vao);
	GLuint sky_vbos[2];
	glGenBuffers(2, sky_vbos);
	glBindBuffer(GL_ARRAY_BUFFER, sky_vbos[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * sky_pos_buf.size(), &sky_pos_buf[0], GL_STATIC_DRAW); //データ転送
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, sizeof(float) * 3, (GLvoid*)0);
	glBindBuffer(GL_ARRAY_BUFFER, sky_vbos[1]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * sky_col_buf.size(), &sky_col_buf[0], GL_STATIC_DRAW); //データ転送
	glEnableClientState(GL_COLOR_ARRAY);
	glColorPointer(3, GL_FLOAT, sizeof(float) * 3, (GLvoid*)0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, pc_vbos[2]); //インデックス配列付ける場合
	//glBufferData(GL_ARRAY_BUFFER, sizeof(float) * pc_indices_buf.size(), &pc_indices_buf[0], GL_STATIC_DRAW); //データ転送
	glBindVertexArray(0);
	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);

  ///////////////////// 右側 //////////////////////////////////
  std::vector<float> right_pos_buf;
  std::vector<float> right_col_buf;

  {
    float right_buf[]={
      //
      200,-5000,-10000, //頂点
      200,-5000,10000,
      200,5000,-1000,
      200,5000,1000,
      // 200,-5000,-100, //頂点
      // 200,-5000,100,
      // 200,5000,-100,
      // 200,5000,100,
      // 100,-5000,-10, //頂点
      // 100,-5000,100,
      // 100,5000,-10,
      // 100,5000,100,

    };
    right_pos_buf.clear(); right_pos_buf.resize(sizeof(right_buf)/sizeof(float));
    memcpy(&right_pos_buf[0],right_buf,sizeof(right_buf));
  }

  // color
  {
    float right_c_buf[]={
      0, 0.5, 0,
      0, 0.5, 0,
      0, 0.5, 0,
      0, 0.5, 0,
    };
    right_col_buf.clear(); right_col_buf.resize(sizeof(right_c_buf)/sizeof(float));
    memcpy(&right_col_buf[0],right_c_buf,sizeof(right_c_buf));
  }

  // VAO
  GLuint right_vao;
	glGenVertexArrays(1, &right_vao);
	glBindVertexArray(right_vao);
	GLuint right_vbos[2];
	glGenBuffers(2, right_vbos);
	glBindBuffer(GL_ARRAY_BUFFER, right_vbos[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * right_pos_buf.size(), &right_pos_buf[0], GL_STATIC_DRAW); //データ転送
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, sizeof(float) * 3, (GLvoid*)0);
	glBindBuffer(GL_ARRAY_BUFFER, right_vbos[1]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * right_col_buf.size(), &right_col_buf[0], GL_STATIC_DRAW); //データ転送
	glEnableClientState(GL_COLOR_ARRAY);
	glColorPointer(3, GL_FLOAT, sizeof(float) * 3, (GLvoid*)0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, pc_vbos[2]); //インデックス配列付ける場合
	//glBufferData(GL_ARRAY_BUFFER, sizeof(float) * pc_indices_buf.size(), &pc_indices_buf[0], GL_STATIC_DRAW); //データ転送
	glBindVertexArray(0);
	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);

  ///////////////////// 左側 //////////////////////////////////
  std::vector<float> left_pos_buf;
  std::vector<float> left_col_buf;

  {
    float left_buf[]={
      //
      -200,-5000,-1000, //頂点
      -200,-5000,10000,
      -200,5000,-1000,
      -200,5000,10000,
      // -120,-5000,-10, //頂点
      // -120,-5000,100,
      // -120,5000,-10,
      // -120,5000,100,

    };
    left_pos_buf.clear(); left_pos_buf.resize(sizeof(left_buf)/sizeof(float));
    memcpy(&left_pos_buf[0],left_buf,sizeof(left_buf));
  }

  // color
  {
    float left_c_buf[]={
      //
      0, 0.5, 0,
      0, 0.5, 0,
      0, 0.5, 0,
      0, 0.5, 0,
    };
    left_col_buf.clear(); left_col_buf.resize(sizeof(left_c_buf)/sizeof(float));
    memcpy(&left_col_buf[0],left_c_buf,sizeof(left_c_buf));
  }

  // VAO
  GLuint left_vao;
	glGenVertexArrays(1, &left_vao);
	glBindVertexArray(left_vao);
	GLuint left_vbos[2];
	glGenBuffers(2, left_vbos);
	glBindBuffer(GL_ARRAY_BUFFER, left_vbos[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * left_pos_buf.size(), &left_pos_buf[0], GL_STATIC_DRAW); //データ転送
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, sizeof(float) * 3, (GLvoid*)0);
	glBindBuffer(GL_ARRAY_BUFFER, left_vbos[1]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * left_col_buf.size(), &left_col_buf[0], GL_STATIC_DRAW); //データ転送
	glEnableClientState(GL_COLOR_ARRAY);
	glColorPointer(3, GL_FLOAT, sizeof(float) * 3, (GLvoid*)0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, pc_vbos[2]); //インデックス配列付ける場合
	//glBufferData(GL_ARRAY_BUFFER, sizeof(float) * pc_indices_buf.size(), &pc_indices_buf[0], GL_STATIC_DRAW); //データ転送
	glBindVertexArray(0);
	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);

  /////////////////////////// transration読み込み ///////////////////////////////
  std::vector<glm::vec3> translist;
  std::vector<glm::vec3> rotatelist;
  std::vector<glm::mat4> extrinsics;
  std::string key; // 画像が 0.png, 1.png, ・・・ 100.png ・・・　のように桁合わせされていない時に使用
  int key_num = 0;
  std::vector<glm::mat4> mat_model_cams; // カメラ軌道確認用 四角錐
  {
    std::fstream ifsjson(dir_name + "reconstruction.json"); //ファイル読み込み
    if(ifsjson.fail()) {
      std::cout << "reconstruction　file read Failed" << std::endl;
      return -1;
    } else {
      std::cout << "Reading reconstruction.json" << std::endl;
    }

    // Picojsonへ読み込む
    picojson::value val;
    ifsjson >> val;
    // fs変数はもう使わないので閉鎖
    ifsjson.close();

    picojson::object& obj = val.get<picojson::array>()[0].get<picojson::object>();
    picojson::object& shots = obj["shots"].get<picojson::object>();
    for (picojson::object::iterator i = shots.begin(); i != shots.end(); ++i)
    {
      // std::cout << i -> first << std::endl;
      key = std::to_string(key_num) + ".png";
      std::cout << key << std::endl;

      picojson::array& trans = i->second.get<picojson::object>()["translation"].get<picojson::array>();
      // picojson::array& trans = shots[key].get<picojson::object>()["translation"].get<picojson::array>();

      translist.push_back(glm::vec3((float)trans[0].get<double>(), (float)trans[1].get<double>(), (float)trans[2].get<double>()));
      std::cout << "x_t:" << translist.back()[0] << std::endl;
      std::cout << "y_t:" << translist.back()[1] << std::endl;
      std::cout << "z_t:" << translist.back()[2] << std::endl;
      //std::cout << "x_t:" << trans[0].get<double>() << std::endl;
      //std::cout << "y_t:" << trans[1].get<double>() << std::endl;
      //std::cout << "z_t:" << trans[2].get<double>() << std::endl;

      picojson::array& rotas = i->second.get<picojson::object>()["rotation"].get<picojson::array>();
      // picojson::array& rotas = shots[key].get<picojson::object>()["rotation"].get<picojson::array>();
      rotatelist.push_back(glm::vec3((float)rotas[0].get<double>(), (float)rotas[1].get<double>(), (float)rotas[2].get<double>()));
      std::cout << "x_r:" << rotatelist.back()[0] << std::endl;
      std::cout << "y_r:" << rotatelist.back()[1] << std::endl;
      std::cout << "z_r:" << rotatelist.back()[2] << std::endl;
      //std::cout << "x_r:" << rotas[0].get<double>() << std::endl;
      //std::cout << "y_r:" << rotas[1].get<double>() << std::endl;
      //std::cout << "z_r:" << rotas[2].get<double>() << std::endl;

      float length = glm::length(rotatelist.back());

      extrinsics.push_back(
       glm::translate(glm::mat4(1.0f), translist.back()) * glm::rotate(length, glm::normalize(rotatelist.back()))
      );

      // カメラの動きを固定カメラから見る時のカメラモデル行列
      // mat_model_cams.push_back(glm::inverse(extrinsics.back()));


      // key_num++;
    }
  }
  std::cout << "Finish reading ply" << std::endl;


	//点群のデータを示すVAOの作成
	GLuint pc_vao;
	glGenVertexArrays(1, &pc_vao);
	glBindVertexArray(pc_vao);
	GLuint pc_vbos[2];
	glGenBuffers(2, pc_vbos);
	glBindBuffer(GL_ARRAY_BUFFER, pc_vbos[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * pc_position_buf.size(), &pc_position_buf[0], GL_STATIC_DRAW); //データ転送
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, sizeof(float) * 3, (GLvoid*)0);
	glBindBuffer(GL_ARRAY_BUFFER, pc_vbos[1]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * pc_color_buf.size(), &pc_color_buf[0], GL_STATIC_DRAW); //データ転送
	glEnableClientState(GL_COLOR_ARRAY);
	glColorPointer(3, GL_FLOAT, sizeof(float) * 3, (GLvoid*)0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, pc_vbos[2]); //インデックス配列付ける場合
	//glBufferData(GL_ARRAY_BUFFER, sizeof(float) * pc_indices_buf.size(), &pc_indices_buf[0], GL_STATIC_DRAW); //データ転送
	glBindVertexArray(0);
	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);

	// 固定機能シェーダを使う場合は行列系のGL関数を使って設定
  // 固定カメラ 上空視点 (注視点は(0,0,0)になっているので、CloudCompareで予め（0,0,0の場所を確認しておくこと）)
	// glm::vec3 camPos(0, 0, 1000); // カメラの座標
	// glm::vec3 gazePos(0.0, 0.0, 0.0); // 注視点
	// glm::vec3 upDir(0.0, 1.0, 0.0); // y軸を"上"と定義
  //
	// glm::mat4 mat_proj = glm::perspective(45.0, (double)dwindowWidth / dwindowHeight, 0.01, 2000.0);
	// glm::mat4 mat_fixedview = glm::lookAt(camPos, gazePos, upDir); // mat_view(LookAt)をextirc paraに変える
	// glm::mat4 mat_model(1.0f);
  // glm::mat4 mat_vm = mat_fixedview * mat_model;

  //64 = 画角
  // xtion, astra だと 49.4がちょうどいい
  // kitti だと 44.47
  glm::mat4 mat_proj = glm::perspective(44.47, (double)dwindowWidth/dwindowHeight, 0.01, 2000.0);
  glm::mat4 mat_view = extrinsics[0];
	glm::mat4 mat_model(1.0f);
  glm::mat4 mat_vm = mat_view * mat_model;

	//行列設定
	glMatrixMode(GL_PROJECTION);
	glLoadMatrixf(glm::value_ptr(mat_proj)); //指定したメモリで行列を埋める．掛け算ではない

	glMatrixMode(GL_MODELVIEW);
	glLoadMatrixf(glm::value_ptr(mat_vm)); //指定したメモリで行列を埋める．掛け算ではない

	//その他設定
	glViewport(0, 0, dwindowWidth, dwindowHeight);
	glClearColor(0, 0, 0, 0); // 背景色
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_PROGRAM_POINT_SIZE); glPointSize(2); //点のサイズの設定．

	SDL_GL_SetSwapInterval(1); //VSYNC使う

	//メインループ
	bool loopFlg = true;
  int fcount=0;
  degree = 0.0;
	while (loopFlg) {

    // std::cout << degree << std::endl;
		//初期化
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


    // 点群を描写するフレーム
    int frame_num = fcount % extrinsics.size();

    // カメラの撮影を点群内で再現
    // 行列適用
    mat_view = extrinsics[frame_num];
    auto x_rotate = glm::rotate(glm::mat4(), (float)glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f) ); // カメラをx軸に180°回転
    mat_view = x_rotate * mat_view;
    auto y_rotate = glm::rotate(glm::mat4(), (float)glm::radians(degree), glm::vec3(0.0f, 1.0f, 0.0f) ); // カメラをy軸に60°回転
    mat_view = y_rotate * mat_view;
    // auto z_rotate = glm::rotate(glm::mat4(), (float)glm::radians(180.0f), glm::vec3(0.0f, 0.0f, 1.0f) ); // カメラをz軸に180°回転
    // mat_view = z_rotate * mat_view;
    glm::mat4 mat_vm = mat_view * mat_model;


    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixf(glm::value_ptr(mat_vm));

    //点群を描画
		glBindVertexArray(pc_vao);
		glDrawArrays(GL_POINTS, 0, pc_position_buf.size() / 3); //描画指示
		glBindVertexArray(0);
    glRotatef(7.0f, 0 ,1, 0);

    ////////////////////////////////////////////////////////////////////////////

//     //床の描写
//     glBindVertexArray(ground_vao); // 有効化
// 		glDrawArrays(GL_QUADS, 0, ground_pos_buf.size() / 3); //描画指示
// 		glBindVertexArray(0);
//
//     //空の描写
//     glBindVertexArray(sky_vao); // 有効化
// 		glDrawArrays(GL_QUADS, 0, sky_pos_buf.size() / 3); //描画指示
// 		glBindVertexArray(0);
// //
//     // 右の描写
//     glBindVertexArray(right_vao); // 有効化
// 		glDrawArrays(GL_QUADS, 0, right_pos_buf.size() / 3); //描画指示
// 		glBindVertexArray(0);
//
//
//     //左の描写
//     glBindVertexArray(left_vao); // 有効化
// 		glDrawArrays(GL_QUADS, 0, left_pos_buf.size() / 3); //描画指示
// 		glBindVertexArray(0);

    // camera 軌道
    // glm::mat4 mat_camera_model = mat_model_cams[frame_num];
    // mat_vm = mat_fixedview * mat_camera_model;

    // glMatrixMode(GL_MODELVIEW);
    // glLoadMatrixf(glm::value_ptr(mat_vm));
    //
    // glBindVertexArray(cam_vao);
		// glDrawArrays(GL_LINES, 0, cube0_pos_buf.size() / 3); //描画指示
		// glBindVertexArray(0);

		//描画完了
		glFlush();

		//描画をウィンドウに反映
		SDL_GL_SwapWindow(window);
    SDL_Delay(100); //delay

    //描画ウィンドウを撮影　
    //最初の15フレーム、最後の残り15フレームは切り捨て
    if (frame_num > 15 && frame_num < extrinsics.size()-15 ) {
      saveImage(width, height, frame_num, dir_name, dataset_name, int(degree));
    }


		//イベント処理
		SDL_Event event;
		while (SDL_PollEvent(&event)) {
			if (event.type == SDL_KEYDOWN) {
				switch (event.key.keysym.sym) {
				case SDLK_ESCAPE:
				case SDLK_q:
					loopFlg = false;
					break;
				}
			}
		}

    fcount++;
    if (fcount % extrinsics.size() == 0  ) {
      return 0;
      // degree += 30.0; //30°ずつ撮影
      if (fcount ==  extrinsics.size()*5 ) {
        return 0;
      }
    }
	}


	return 0;
}
