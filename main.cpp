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
// Nev    : Barta Gergo
// Neptun : YM4TZ1
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
GLuint shaderProg;
constexpr int t_level_u = 30;
constexpr int t_level_v = 30;
constexpr float G = 0.1f;

void getErrorInfo(unsigned int handle) {
    int logLen;
    glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
    if (logLen > 0) {
        char * log = new char[logLen];
        int written;
        glGetShaderInfoLog(handle, logLen, &written, log);
        printf("Shader log:\n%s", log);
        delete log;
    }
}

// check if shader could be compiled
void checkShader(unsigned int shader, char * message) {
    int OK;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
    if (!OK) {
        printf("%s!\n", message);
        getErrorInfo(shader);
    }
}

// check if shader could be linked
void checkLinking(unsigned int program) {
    int OK;
    glGetProgramiv(program, GL_LINK_STATUS, &OK);
    if (!OK) {
        printf("Failed to link shader program!\n");
        getErrorInfo(program);
    }
}

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

    void SetUniform(GLuint shaderProg, const char * name) {
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

    static vec4 cross(const vec4& v1, const vec4& v2) {
		return vec4(v1.v[1] * v2.v[2] - v1.v[2] * v2.v[1],
					v1.v[2] * v2.v[0] - v1.v[0] * v2.v[2],
					v1.v[0] * v2.v[1] - v1.v[1] * v2.v[0], 0.0f);
	}

    void SetUniform4(GLuint shaderProg, const char *name) {
		int loc = glGetUniformLocation(shaderProg, name);
		glUniform4f(loc, v[0], v[1], v[2], v[3]);
	}

    void SetUniform3(GLuint shaderProg, const char *name) {
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
    vec4 v, g;
    float m, acc, l_acc;
    Light(const vec4& pos, const vec4& le, const vec4& la/*, vec4 v0, float m*/)
        : position(pos), La(la), Le(le)/*, v(v0), m(m), acc(0), l_acc(0), g(0, -1, 0)*/ { }

  //  vec4 getDir(vec4 g) {
  //      return v - g * (acc - l_acc);
  //  }

    void Animate(float dt);
};

struct RenderState {
    mat4 M, V, P, Minv;
    Material *material;
    Light *light[2];
    vec4 wEye;
};

struct Shader {
    void Create(const char * vsSrc, const char * vsAttrNames[], int size,
                const char * fsSrc, const char * fsOuputName) {
        unsigned int vs = glCreateShader(GL_VERTEX_SHADER);
        if(!vs) {
            printf("Error in vertex shader creation\n");
            exit(1);
        }
        glShaderSource(vs, 1, &vsSrc, NULL); glCompileShader(vs);
        checkShader(vs, "Vertex shader error");
        unsigned int fs = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fs, 1, &fsSrc, NULL); glCompileShader(fs);
        if (!fs) {
            printf("Error in fragment shader creation\n");
            exit(1);
        }
        shaderProg = glCreateProgram();
        if (!shaderProg) {
            printf("Error in shader program creation\n");
            exit(1);
        }
        glAttachShader(shaderProg, vs);
        glAttachShader(shaderProg, fs);
        for (int i = 0; i < size; i++)
            glBindAttribLocation(shaderProg, i, vsAttrNames[i]);
        glBindFragDataLocation(shaderProg, 0, fsOuputName);
        glLinkProgram(shaderProg);
        checkLinking(shaderProg);
    }

    virtual void Bind(RenderState *state) = 0;
};

class PerPixelShader : public Shader {
const char *vsSrc = R"(
#version 130
precision highp float;

uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
uniform vec4  wLiPos_a, wLiPos_b;       // pos of light source
uniform vec3  wEye;         // pos of eye

in vec4 vtxPos;            // pos in modeling space
in vec4 vtxNorm;           // normal in modeling space
in vec2 vtxUV;

out vec2 texcoord;
out vec3 wNormal;           // normal in world space
out vec3 wView;             // view in world space
out vec3 wLight_a;            // light dir in world space
out vec3 wLight_b;

void main() {
   gl_Position = vec4(vtxPos.xyz, 1) * MVP; // to NDC
   texcoord = vtxUV;
   vec4 wPos = vec4(vtxPos.xyz, 1) * M;
   wLight_a  = wLiPos_a.xyz * wPos.w - wPos.xyz * wLiPos_a.w;
   wLight_b  = wLiPos_b.xyz * wPos.w - wPos.xyz * wLiPos_b.w;
   wView   = wEye * wPos.w - wPos.xyz;
   wNormal = (Minv * vec4(vtxNorm.xyz, 0)).xyz;
})";

const char *fsSrc = R"(
#version 130
precision highp float;

uniform vec4 kd, ks, ka;// diffuse, specular, ambient ref
uniform vec4 La_a, Le_a, La_b, Le_b;    // ambient and point source rad
uniform float shine;    // shininess for specular ref
uniform sampler2D samplerUnit;

in vec2 texcoord;
in  vec3 wNormal;       // interpolated world sp normal
in  vec3 wView;         // interpolated world sp view
in  vec3 wLight_a;        // interpolated world sp illum dir
in  vec3 wLight_b;
out vec4 fragmentColor; // output goes to frame buffer

void main() {
   vec3 N = normalize(wNormal);
   vec3 V = normalize(wView);
   vec3 L_a = normalize(wLight_a);
   vec3 L_b = normalize(wLight_b);
   vec3 H_a = normalize(L_a + V);
   vec3 H_b = normalize(L_b + V);
   float cost_a = max(dot(N,L_a), 0), cosd_a = max(dot(N,H_a), 0);
   float cost_b = max(dot(N,L_b), 0), cosd_b = max(dot(N,H_b), 0);
   vec3 color = ka.xyz * (La_a.xyz + La_b.xyz) +
               (kd.xyz * (cost_a + cost_b) + ks.xyz * pow((cosd_a + cosd_b), shine)) * (Le_a.xyz + Le_a.xyz);
   fragmentColor = vec4(color, 1) + texture(samplerUnit, texcoord);
}
)";
public:
    PerPixelShader() { }

    void init() {
        static const char *vsAttrNames[] = { "vtxPos", "vtxNorm" };
        Create(vsSrc, vsAttrNames, 2, fsSrc, "fragmentColor");
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
        state->light[0]->La.SetUniform4(shaderProg, "La_a");
        state->light[0]->Le.SetUniform4(shaderProg, "Le_a");
        state->light[0]->position.SetUniform4(shaderProg, "wLiPos_a");
        state->light[1]->La.SetUniform4(shaderProg, "La_b");
        state->light[1]->Le.SetUniform4(shaderProg, "Le_b");
        state->light[1]->position.SetUniform4(shaderProg, "wLiPos_b");
        state->wEye.SetUniform3(shaderProg, "wEye");
        int loc = glGetUniformLocation(shaderProg, "shine");
        glUniform1f(loc, state->material->shine);
    }
};

PerPixelShader sh;

struct Texture {
   unsigned int textureId;
   int width, height;
   float *image;

   Texture(int w, int h) : height(h), width(w) {
       image = new float[w * h * 3];
   }

   void LoadImage(int w, int h) {
       float *img = image;
       for(int i=0; i<w; i++) {
           for(int j=0; j<h; j++) {
               *img++ = floorf(fabs(sinf(i)) + 0.5f) * 0.4f;
               *img++ = floorf(fabs(sinf(j)) + 0.5f) * 0.2f;
               *img++ = 0.0f;
           }
       }
   }

   void init() {
      glGenTextures(1, &textureId);
      glBindTexture(GL_TEXTURE_2D, textureId);
      LoadImage(width, height);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height,
                   0, GL_RGB, GL_FLOAT, image);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  }
};

Texture texture(100, 100);

struct Geometry {
    unsigned int vao, nVtx;

    void init() {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
    }

    void Draw(Shader *sh) {
        int samplerUnit = 0;
        int location = glGetUniformLocation(shaderProg, "samplerUnit");
        glUniform1i(location, samplerUnit);
        glActiveTexture(GL_TEXTURE0 + samplerUnit);
        glBindTexture(GL_TEXTURE_2D, texture.textureId);
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, nVtx);
    }
};

struct VertexData {
    vec4 position, normal;
    float u, v;
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

        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, stride, (void*)0);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, stride, (void*)sVec4);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, (void*)(2 * sVec4));
        delete[] vtxData;
    }
};

class Sphere : public ParamSurface {
    vec4 center;
    float radius;
public:
    Sphere(vec4 c, float r) : center(c), radius(r) { }

    void init_sphere() {
        Create(t_level_u, t_level_v);
    }

    VertexData GenVertexData(float u, float v) {
        VertexData vd;
        vd.normal = vec4(cosf(u * 2.0f * M_PI) * sinf(v * M_PI),
                         sinf(u * 2.0f * M_PI) * sinf(v * M_PI),
                         cosf(v * M_PI), 0.0f).normalize();
        vd.position = vd.normal * radius + center;
        vd.u = u; vd.v = v;
        return vd;
    }
};

class Torus : public ParamSurface {
    vec4 center;
    float r, R;
public:
    Torus(vec4 c, float r, float R) : center(c), r(r), R(R) { }

    void init_torus() {
        Create(t_level_u, t_level_v); // tessellation level
    }

    VertexData drdt(float ut, float vt) {

    }

    VertexData GenVertexData(float u, float v) {
        VertexData vd;
        vd.position = vec4((R + r * cosf(2.0f * M_PI * u)) * cosf(2.0f * M_PI * v),
                           r * sinf(2.0f * M_PI * u),
                           (R + r * cosf(2.0f * M_PI * u)) * sinf(2.0f * M_PI * v));
        vd.normal = vec4(-(cosf(2.0f * M_PI * u) * cosf(2.0f * M_PI * v)),
                         -(sinf(2.0f * M_PI * u)),
                         -(cosf(2.0f * M_PI * u) * sinf(2.0f * M_PI * v))).normalize();
        vd.u = u; vd.v = v;
        return vd;
    }
};

struct Object {
	Shader *shader;
	Material *material;
	Geometry *geometry;
	Camera *camera;
    vec4 scale, pos, rotAxis;
    mat4 Manim, Maniminv;
	float rotAngle;

    Object(Shader *shader, Material *material, Geometry *geometry, Camera *camera,
		   const vec4& scale, const vec4& pos, const vec4& rotAxis, float rotangle)
			: shader(shader), material(material), geometry(geometry), scale(scale), pos(pos), rotAxis(rotAxis),
              rotAngle(rotangle), camera(camera) {
        Manim = mat4(1, 0, 0, 0,
                     0, 1, 0, 0,
                     0, 0, 1, 0,
                     0, 0, 0, 1);
        Maniminv = mat4(1, 0, 0, 0,
                     0, 1, 0, 0,
                     0, 0, 1, 0,
                     0, 0, 0, 1);
    }
    mat4 M() {
        return mat4::Scale(scale.v[0], scale.v[1], scale.v[2]) *
               mat4::Rotate(rotAngle, rotAxis.v[0], rotAxis.v[1], rotAxis.v[2]) * Manim *
               mat4::Translate(pos.v[0], pos.v[1], pos.v[2]);
    }

	virtual void Draw(RenderState *state) {
        state->M = M();
        state->Minv = mat4::Translate(-pos.v[0], -pos.v[1], -pos.v[2]) * Maniminv *
                      mat4::Rotate(-rotAngle,rotAxis.v[0],rotAxis.v[1],rotAxis.v[2]) *
					  mat4::Scale(1.0f / scale.v[0], 1.0f / scale.v[1], 1.0f / scale.v[2]);
		state->material = material;
		state->wEye = camera->wEye;
		shader->Bind(state);
        geometry->Draw(shader);
	}

    virtual void Animate(float dt) = 0;
};

struct TorusObject : public Object {
    TorusObject(Shader *shader, Material *material, Geometry *geometry, Camera *camera,
                 const vec4& scale, const vec4& pos, const vec4& rotAxis, float rotangle)
        : Object(shader, material, geometry, camera, scale, pos, rotAxis, rotangle) { }

    void Animate(float dt)  {

    }
};

Camera camera(vec4(0, -5, 15), vec4(0, 0, 0), vec4(0, 1.0f, 0), (float)M_PI / 2.0f, 1.0f, 1.0f, 100.0f);
Sphere sphere(vec4(0, 0, 0), 1.0f);
Torus torus(vec4(0, 0, 0), 0.8f, 1.0f);
Material jade(vec4(0.54f, 0.89f, 0.63f), vec4(0.316228f, 0.316228f, 0.316228f), vec4(0.135f, 0.2225f, 0.1575f), 0.1);
Light light1(vec4(0, 30, 40), vec4(0, 0.2, 0.2), vec4(0.3, 0.3, 0.3));
Light light2(vec4(0, 25, 33), vec4(0.2, 0.2, 0), vec4(0.3, 0.3, 0.3));
TorusObject oTorus(&sh, &jade, &torus, &camera, vec4(10, 10, 10), vec4(0, 0, 0), vec4(0, 0, 1), M_PI / 5.0f);

void Light::Animate(float dt) {


}

struct SphereObject : public Object {
    Torus *torus; TorusObject *to;
    float acc;
    vec4 l_pos;

    SphereObject(Shader *shader, Material *material, Geometry *geometry, Camera *camera,
                 const vec4& scale, const vec4& pos, const vec4& rotAxis, float rotangle, Torus *torus, TorusObject *to)
        : torus(torus), to(to), acc(0), l_pos(), Object(shader, material, geometry, camera, scale, pos, rotAxis, rotangle) { }

    VertexData getTorusPoint() {
        float u = cosf(acc);
        float v = cosf(acc);
        return torus->GenVertexData(u, v);
    }

    void Animate(float dt)  {
        dt = dt / 20.0f;
        acc += dt;
        VertexData vd = getTorusPoint();
        vd.position = vd.position * to->M();
        vd.normal = (vd.normal * to->M()).normalize();

        pos = (vd.position + vd.normal * 1.2f);
        vec4 dir = (pos - l_pos).normalize();
        vec4 rAxis = vec4::cross(vd.normal, dir).normalize();

        float rot = -dir.length() * 80 * dt / -(vd.normal.length() * 1.2f);
        Manim = Manim * mat4::Rotate(-rot, rAxis.v[0], rAxis.v[1], rAxis.v[2]);
        Maniminv = mat4::Rotate(rot, rAxis.v[0], rAxis.v[1], rAxis.v[2]) * Maniminv;
        l_pos = pos;
    }
};

SphereObject oSphere(&sh, &jade, &sphere, &camera, vec4(1.2, 1.2, 1.2), vec4(0, 0, 10), vec4(0, 0, 1), 0.0f, &torus, &oTorus);

class Scene {
	Camera *camera;
	Object *objects[100];
    unsigned int nObj, nLights;
    Light *light[2];
	RenderState state;
public:
    Scene(Camera *camera) : camera(camera), state(), nObj(0), nLights(0) {}

    void Render() {
		state.wEye = camera->wEye;
        state.V = camera->V();
		state.P = camera->P();
        for(int j = 0; j < nLights; j++) state.light[j] = light[j];
		for(int i = 0; i < nObj; i++) objects[i]->Draw(&state);
	}

    void addObject(Object *o) { objects[nObj++] = o; }

    void addLight(Light *l) { light[nLights++] = l; }

    void Animate(float dt) {
        for(int i = 0; i < nLights; i++) light[i]->Animate(dt);
        for(int j = 0; j < nObj; j++) objects[j]->Animate(dt);
    }
};

Scene scene(&camera);

// Initialization, create an OpenGL context
void onInitialization() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST); // z-buffer is on
    glDisable(GL_CULL_FACE); // backface culling is off
	glViewport(0, 0, windowWidth, windowHeight);
    sh.init();
    texture.init();
    torus.init();
    torus.init_torus();
    sphere.init();
    sphere.init_sphere();
    scene.addObject(&oTorus);
    scene.addObject(&oSphere);
    scene.addLight(&light1);
    scene.addLight(&light2);
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
    static float tend = 0;
    const float dt = 0.01;
    float tstart = tend;
    tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
    for(float t = tstart; t < tend; t += dt) {
        float Dt = fmin(dt, (tend - t));
        scene.Animate(Dt);
    }
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

