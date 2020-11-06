#version 450
layout (location = 0) in vec2 inUV0;
layout (location = 1) in vec2 inUV1;
layout (location = 2) in vec3 inPos;
layout (location = 3) in vec3 inNormal;
layout (location = 4) in vec3 inColor;
layout (location = 5) in vec4 inTangent;

layout (set = 0, binding = 0) uniform UBOScene {
	mat4 projection;
	mat4 view;
	mat4 model;
	vec4 lightPos;
	vec4 viewPos;
} uboScene;

layout (location = 0) out vec2 outUV0;
layout (location = 1) out vec2 outUV1;
layout (location = 2) out vec3 outNormal;
layout (location = 3) out vec3 outColor;
layout (location = 4) out vec4 outTangent;

layout (location = 5) out vec3 outViewVec;
layout (location = 6) out vec3 outLightVec;

void main()  {
    outUV0 = inUV0;
	outUV1 = inUV0;
	outNormal = (transpose(inverse(uboScene.model)) * vec4(inNormal,1.0)).xyz;
	outColor = inColor;
    outTangent = inTangent;

	gl_Position = uboScene.projection * uboScene.view * uboScene.model * vec4(inPos.xyz, 1.0);
    
	vec4 pos = uboScene.model * vec4(inPos, 1.0);
	outLightVec = uboScene.lightPos.xyz - pos.xyz;
	outViewVec = uboScene.viewPos.xyz - pos.xyz;
}