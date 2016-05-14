//=============================================================================================
// Szamitogepes grafika hazi feladat keret. Ervenyes 2016-tol.
// A //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sorokon beluli reszben celszeru garazdalkodni, mert a tobbit ugyis toroljuk.
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiv�ve
// - new operatort hivni a lefoglalt adat korrekt felszabaditasa nelkul
// - felesleges programsorokat a beadott programban hagyni
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : 
// Neptun : 
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <vector>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		// must be downloaded 
#include <GL/freeglut.h>	// must be downloaded unless you have an Apple
#endif

const unsigned int windowWidth = 600, windowHeight = 600;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Innentol modosithatod...

// OpenGL major and minor versions
int majorVersion = 3, minorVersion = 0;

// row-major matrix 4x4
struct mat4 {
	float m[4][4];
public:
	mat4() {}
	mat4(float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33) {
		m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
		m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
		m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
		m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
	}

	void SetUniform(unsigned shaderProg, const char * name) {
		int loc = glGetUniformLocation(shaderProg, name);
		glUniformMatrix4fv(loc, 1, GL_TRUE, &m[0][0]);
	}

	static mat4 Translate(float tx, float ty, float tz) {
		return mat4(1,   0,   0,   0,
					0,   1,   0,   0,
					0,   0,   1,   0,
					tx,  ty,  tz,  1);
	}

	static mat4 Scale(float sx, float sy, float sz) {
		return mat4(sx,   0,    0,   0,
					 0,  sy,    0,   0,
					 0,   0,   sz,   0,
					 0,   0,    0,   1);
	}

	static mat4 Rotate(float angle,float wx,float wy,float wz);

	mat4 operator*(const mat4& right) {
		mat4 result;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				result.m[i][j] = 0;
				for (int k = 0; k < 4; k++) result.m[i][j] += m[i][k] * right.m[k][j];
			}
		}
		return result;
	}
	operator float*() { return &m[0][0]; }
};

// 3D point in homogeneous coordinates
struct vec4 {
	float v[4];

	vec4(float x = 0, float y = 0, float z = 0, float w = 1) {
		v[0] = x; v[1] = y; v[2] = z; v[3] = w;
	}

	vec4 operator*(float f) {
		return vec4(v[0] * f, v[1] * f, v[2] * f, v[3] * f);
	}

	vec4 operator+(const vec4& vec) {
		return vec4(v[0] + vec.v[0], v[1] + vec.v[1], v[2] + vec.v[2], v[3] + vec.v[3]);
	}

	vec4 operator-(const vec4& vec) {
		return vec4(v[0] - vec.v[0], v[1] - vec.v[1], v[2] - vec.v[2], v[3] - vec.v[3]);
	}

	vec4 operator*(const mat4& mat) {
		vec4 result;
		for (int j = 0; j < 4; j++) {
			result.v[j] = 0;
			for (int i = 0; i < 4; i++) result.v[j] += v[i] * mat.m[i][j];
		}
		return result;
	}

	float length() {
		return sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
	}

	vec4 normalize() {
		return *this * (1.0f / length());
	}

	static float dot(vec4 v1, vec4 v2) {
		return (v1.v[0] * v2.v[0] + v1.v[1] * v2.v[1] + v1.v[2] * v2.v[2]);
	}

	static vec4 cross(vec4 v1, vec4 v2) {
		return vec4(v1.v[1] * v2.v[2] - v1.v[2] * v2.v[1],
					v1.v[2] * v2.v[0] - v1.v[0] * v2.v[2],
					v1.v[0] * v2.v[1] - v1.v[1] * v2.v[0], 0.0f);
	}

	void SetUniform4(unsigned shaderProg, const char *name) {
		int loc = glGetUniformLocation(shaderProg, name);
		glUniform4f(loc, v[0], v[1], v[2], v[3]);
	}

	void SetUniform3(unsigned shaderProg, const char *name) {
		int loc = glGetUniformLocation(shaderProg, name);
		glUniform3f(loc, v[0], v[1], v[2]);
	}
};

mat4 mat4::Rotate(float angle, float wx, float wy, float wz) {
	vec4 wxyz(wx, wy, wz, 0); wxyz.normalize();
	float x = wxyz.v[0]; float y = wxyz.v[1]; float z = wxyz.v[2];
	float c = cosf(angle); float s = sinf(angle);
	return mat4(x * x * (1.0f - c) + c, x * y * (1.0f - c) - z * s, x * z * (1.0f - c) + y * s, 0,
				y * x * (1.0f - c) + z * s, y * y * (1.0f - c) + c, y * z * (1.0f - c) - x * s, 0,
				x * z * (1.0f - c) - y * s, y * z * (1.0f - c) + x * s, z * z * (1.0f - c) + c,	0,
				0,							0,							0,						1);
}

struct Geometry {
	unsigned int vao, nVtx;

	void init() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
	}

	void Draw() {
		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLES, 0, nVtx);
	}
};

struct VertexData {
	vec4 position, normal;
	//float u, v;
};

struct ParamSurface : public Geometry {
    ParamSurface() : Geometry() {}

	virtual VertexData GenVertexData(float u, float v) = 0;

	void Create(int N, int M) {
		nVtx = N * M * 6;
		unsigned int vbo;
		glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);

		VertexData *vtxData = new VertexData[nVtx], *pVtx = vtxData;
		for (int i = 0; i < N; i++) for (int j = 0; j < M; j++) {
				*pVtx++ = GenVertexData((float)i / N, (float)j / M);
				*pVtx++ = GenVertexData((float)(i + 1) / N, (float)j / M);
				*pVtx++ = GenVertexData((float)i / N, (float)(j + 1) / M);
				*pVtx++ = GenVertexData((float)(i + 1) / N, (float)j / M);
				*pVtx++ = GenVertexData((float)(i + 1) / N, (float)(j + 1) / M);
				*pVtx++ = GenVertexData((float)i / N, (float)(j + 1) / M);
			}

		int stride = sizeof(VertexData), sVec4 = sizeof(vec4);
		glBufferData(GL_ARRAY_BUFFER, nVtx * stride, vtxData, GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);  // AttribArray 0 = POSITION
		glEnableVertexAttribArray(1);  // AttribArray 1 = NORMAL
		//glEnableVertexAttribArray(2);  // AttribArray 2 = UV
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, stride, (void*)0);
		glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, stride, (void*)sVec4);
		//glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, (void*)(2 * sVec4));
	}
};

class Sphere : public ParamSurface {
	vec4 center;
	float radius;
public:
	Sphere(vec4 c, float r) : center(c), radius(r) { }

	void init_sphere() {
		Create(30, 30); // tessellation level
	}

	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		vd.normal = vec4(cosf(u * 2.0f * M_PI) * sinf(v * M_PI),
						 sinf(u * 2.0f * M_PI) * sinf(v * M_PI),
						 cosf(v * M_PI), 0.0f).normalize();
		vd.position = vd.normal * radius + center;
		//vd.u = u; vd.v = v;
		return vd;
	}
};

struct Camera {
	vec4  wEye, wLookat, wVup;
	float fov, asp, fp, bp;
    Camera(const vec4& eye, const vec4& lookat, const vec4& up,
           float fov, float asp, float fp, float bp) : wEye(eye), wLookat(lookat), wVup(up),
                                                       fov(fov), asp(asp), fp(fp), bp(bp) {}

	mat4 V() { // view matrix
		vec4 w = (wEye - wLookat).normalize();
		vec4 u = vec4::cross(wVup, w).normalize();
		vec4 v = vec4::cross(w, u);
		return mat4::Translate(-wEye.v[0], -wEye.v[1], -wEye.v[2]) *
			   mat4(u.v[0],  v.v[0],  w.v[0],  0.0f,
					u.v[1],  v.v[1],  w.v[1],  0.0f,
					u.v[2],  v.v[2],  w.v[2],  0.0f,
					0.0f,    0.0f,    0.0f,    1.0f );
	}
	mat4 P() { // projection matrix
		float sy = 1.0f / tanf(fov / 2.0f);
		return mat4(sy/asp, 0.0f,  0.0f,                        0.0f,
					0.0f,   sy,    0.0f,                        0.0f,
					0.0f,   0.0f, -(fp + bp) / (bp - fp),      -1.0f,
					0.0f,   0.0f, -2.0f * fp * bp / (bp - fp),  0.0f);
	}
};

struct Material {
	vec4 kd, ks, ka;
	float shine;
	Material(const vec4& kd, const vec4& ks, const vec4& ka, float shine) : kd(kd), ks(ks), ka(ka), shine(shine) { }
};

struct Light {
	vec4 position;
	vec4 La, Le;
	Light(const vec4& pos, const vec4& le, const vec4& la) : position(pos), La(la), Le(le) { }
};

struct RenderState {
	mat4 M, V, P, Minv;
	Material *material;
	Light *light;
	vec4 wEye;
};

struct Shader {
	unsigned int shaderProg;

	void Create(const char * vsSrc, const char * vsAttrNames[],
				const char * fsSrc, const char * fsOuputName) {
		unsigned int vs = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vs, 1, &vsSrc, NULL); glCompileShader(vs);
		unsigned int fs = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fs, 1, &fsSrc, NULL); glCompileShader(fs);
		GLuint shaderProgram = glCreateProgram();
		glAttachShader(shaderProg, vs);
		glAttachShader(shaderProg, fs);
		for (int i = 0; i < sizeof(vsAttrNames)/sizeof(char*); i++)
			glBindAttribLocation(shaderProg, i, vsAttrNames[i]);
		glBindFragDataLocation(shaderProg, 0, fsOuputName);
		glLinkProgram(shaderProg);
	}

	virtual void Bind(RenderState *state) {	}
};
/*
class PerPixelShader : public Shader {
const char *vsSrc = R"(
uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
uniform vec4  wLiPos;       // pos of light source
uniform vec3  wEye;         // pos of eye

in  vec4 vtxPos;            // pos in modeling space
in  vec4 vtxNorm;           // normal in modeling space

out vec3 wNormal;           // normal in world space
out vec3 wView;             // view in world space
out vec3 wLight;            // light dir in world space

void main() {
   gl_Position = vec4(vtxPos.xyz, 1) * MVP; // to NDC
   vec4 wPos = vec4(vtxPos.xyz, 1) * M;
   wLight  = wLiPos.xyz * wPos.w - wPos.xyz * wLiPos.w;
   wView   = wEye * wPos.w - wPos.xyz;
   wNormal = (Minv * vec4(vtxNorm.xyz, 0)).xyz;
})";

const char *fsSrc = R"(
uniform vec4 kd, ks, ka;// diffuse, specular, ambient ref
uniform vec4 La, Le;    // ambient and point source rad
uniform float shine;    // shininess for specular ref

in  vec3 wNormal;       // interpolated world sp normal
in  vec3 wView;         // interpolated world sp view
in  vec3 wLight;        // interpolated world sp illum dir
out vec4 fragmentColor; // output goes to frame buffer

void main() {
   vec3 N = normalize(wNormal);
   vec3 V = normalize(wView);
   vec3 L = normalize(wLight);
   vec3 H = normalize(L + V);
   float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
   vec3 color = ka.xyz * La.xyz +
               (kd.xyz * cost + ks.xyz * pow(cosd,shine)) * Le.xyz;
   fragmentColor = vec4(0, 1, 0, 1);
}}
)";
public:
	PerPixelShader() { }

	void create() {
		static const char *vsAttrNames[] = { "vtxPos", "vtxNorm" };
		Create(vsSrc, vsAttrNames, fsSrc, "fragmentColor");
	}

	void Bind(RenderState *state) {
		glUseProgram(shaderProg);
		mat4 MVP = state->M * state->V * state->P;
		MVP.SetUniform(shaderProg, "MVP");
		state->M.SetUniform(shaderProg, "M");
		state->Minv.SetUniform(shaderProg, "Minv");
		state->material->kd.SetUniform4(shaderProg, "kd");
		state->material->ks.SetUniform4(shaderProg, "ks");
		state->material->ka.SetUniform4(shaderProg, "ka");
		state->light->La.SetUniform4(shaderProg, "La");
		state->light->Le.SetUniform4(shaderProg, "Le");
		state->light->position.SetUniform4(shaderProg, "wLiPos");
		state->wEye.SetUniform3(shaderProg, "wEye");
		int loc = glGetUniformLocation(shaderProg, "shine");
		glUniform1f(loc, state->material->shine);
	}
};*/

class ShadowShader : public Shader {
const char * vsSrc = R"(
   uniform mat4 MVP;
   in vec4 vtxPos;
   void main() { gl_Position = vtxPos * MVP; }
)";

const char * fsSrc = R"(
	out vec4 fragmentColor;
	void main() { fragmentColor = vec4(1, 0, 0, 1); }
)";
public:

	ShadowShader() { }

	void init() {
		static const char * vsAttrNames[] = { "vtxPos" };
		Create(vsSrc, vsAttrNames, fsSrc, "fragmentColor");
	}

	void Bind(RenderState& state) {
		glUseProgram(shaderProg);
		mat4 MVP = state.M * state.V * state.P;
		MVP.SetUniform(shaderProg, "MVP");
	}
};

class Object {
	Shader *shader;
	Material *material;
	Geometry *geometry;
	Camera *camera;
	vec4 scale, pos, rotAxis;
	float rotAngle;
public:
	Object(Shader *shader, Material *material, Geometry *geometry, Camera *camera,
		   const vec4& scale, const vec4& pos, const vec4& rotAxis, float rotangle)
			: shader(shader), material(material), geometry(geometry), scale(scale), pos(pos), rotAxis(rotAxis),
			  rotAngle(rotangle), camera(camera) {}

	virtual void Draw(RenderState *state) {
		state->M = mat4::Scale(scale.v[0], scale.v[1], scale.v[2]) *
				   mat4::Rotate(rotAngle, rotAxis.v[0], rotAxis.v[1], rotAxis.v[2]) *
				   mat4::Translate(pos.v[0], pos.v[1], pos.v[2]);
		state->Minv = mat4::Translate(-pos.v[0], -pos.v[1], -pos.v[2]) *
					  mat4::Rotate(-rotAngle,rotAxis.v[0],rotAxis.v[1],rotAxis.v[2]) *
					  mat4::Scale(1.0f / scale.v[0], 1.0f / scale.v[1], 1.0f / scale.v[2]);
		state->material = material;
		state->wEye = camera->wEye;
		shader->Bind(state);
		geometry->Draw();
	}
	// virtual void Animate(float dt) {}
};

class Scene {
	Camera *camera;
	Object *objects[100];
	unsigned int nObj;
	Light *light;
	RenderState state;
public:
    Scene(Camera *camera, Light *light) : camera(camera), light(light), state() {}

    void Render() {
		state.wEye = camera->wEye;
		state.V = camera->V();
		state.P = camera->P();
		state.light = light;
		for(int i = 0; i < nObj; i++) objects[i]->Draw(&state);
	}

    void addObject(Object *o) {
        objects[nObj++] = o;
    }

	/*void Animate(float dt) {
		for (Object * obj : objects) obj->Animate(dt);
	}*/
};

// handle of the shader program
//unsigned int shaderProgram;

Camera camera(vec4(0, 0, 1), vec4(0, 0, 0), vec4(0, 1.0f, 0), (float)M_PI / 2.0f, 1.0f, -100.0f, 100.0f);
Sphere sphere(vec4(0, 0, 0), 0.2f);
Material jade(vec4(0.54f, 0.89f, 0.63f), vec4(0.316228f, 0.316228f, 0.316228f), vec4(0.135f, 0.2225f, 0.1575f), 0.1);
Light light(vec4(20, 20, 50), vec4(1, 1, 1), vec4(0, 0, 0));
ShadowShader sh;
Object oSphere(&sh, &jade, &sphere, &camera, vec4(1, 1, 1), vec4(0, 0, 0), vec4(0, 0, 0), 0.0f);
Scene scene(&camera, &light);

// Initialization, create an OpenGL context
void onInitialization() {
	//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	//glEnable(GL_DEPTH_TEST); // z-buffer is on
	//glDisable(GL_CULL_FACE); // backface culling is off
	glViewport(0, 0, windowWidth, windowHeight);
	sphere.init();
	sphere.init_sphere();
	scene.addObject(&oSphere);

}

void onExit() {
	//glDeleteProgram(shaderProgram);
	//printf("exit");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
	scene.Render();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
		float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
		float cY = 1.0f - 2.0f * pY / windowHeight;
		glutPostRedisplay();     // redraw
	}
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	float sec = time / 1000.0f;				// animate the triangle object
	glutPostRedisplay();					// redraw the scene
}

// Idaig modosithatod...
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

int main(int argc, char * argv[]) {
	glutInit(&argc, argv);
#if !defined(__APPLE__)
	glutInitContextVersion(majorVersion, minorVersion);
#endif
	glutInitWindowSize(windowWidth, windowHeight);				// Application window is initially of resolution 600x600
	glutInitWindowPosition(100, 100);							// Relative location of the application window
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_2_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
	glewExperimental = true;	// magic
	glewInit();
#endif

	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	onInitialization();

	glutDisplayFunc(onDisplay);                // Register event handlers
	glutMouseFunc(onMouse);
	glutIdleFunc(onIdle);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutMotionFunc(onMouseMotion);

	glEnable(GL_DEPTH_TEST); // z-buffer is on
	glDisable(GL_CULL_FACE); // backface culling is off

	glutMainLoop();
	onExit();
	return 1;
}

