---
title: A quick introduction to Direct State Access
author: Robert Ryan
date: 2020-12-09 06:37:00 +0100
tags:
  - graphics
  - opengl
  - optimization
  - azdo
pin: true
---

# A quick introduction to Direct State Access in OpenGL
Direct State Access (*DSA*) is a paradigm and set of functions OpenGL offers that basically allow you to ignore binding resources to change them.

<!--more-->

All of those functions are available as of OpenGL 4.5 core, or if you're using an earlier version of OpenGL,
the [`GL_ARB_direct_state_access`](https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_direct_state_access.txt) extension and even earlier the [`GL_EXT_direct_state_access`](https://www.khronos.org/registry/OpenGL/extensions/EXT/EXT_direct_state_access.txt) extension.
In this introduction I will show the most important aspects of those functions and how they can easily help you to gain performance on the CPU-side, simply by reducing the binding.
In general what was done without DSA was a call to `glGen*` followed by `glBind*`. This will be replaced by `glCreate*`. The signature for `glGen*` and `glCreate*` is the same.
That means, in an optimal world, we will only bind something in the case we want to render it or use it on the GPU for something else, *Compute* for example.

## Buffers
As stated before, `glGen*` and `glBind*` will be replaced by `glCreate*`. So for buffers, it's as simple as calling [`glCreateBuffers`](https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glCreateBuffers.xhtml). The signature is the same as in `glGen`, with the difference that we actually gain an usable buffer. `glGen` only creates *names* which we can then use with `glBind`, which would automatically create them for us.
Now we have a buffer that we can use. To increase performance even further, we only want to allocate memory once and have complete control over it, by making the buffer size constant.
In that case, we call [`glNamedBufferStorage(GLuint buffer, GLsizeiptr size, const void* data, GLbitfield flags)`](https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glBufferStorage.xhtml). `buffer` is the actual ID we generate with `glCreateBuffers`.
`size` is the size we want this buffer to have. `data` is the initial data we want to upload. This can be `nullptr`. `flags` are some flags we can set. This can be `0`, but I'll explain now what this field actually does.
With `flags` we tell the driver how we want to use this buffer. We might want to use it only on the GPU, with no upload from the CPU side whatsoever, then setting it to `0` (no flags) is completely fine and recommended.
If we want to use it as you'd normally expect, with `glNamedBufferData` / `glNamedBufferSubData` (DSA-equivalent to `glBufferData` / `glBufferSubData`), then we'll have to set the `GL_DYNAMIC_STORAGE_BIT`.
There are other flags we can set, but for now, we won't care about those flags. But if you're interested, they are `GL_MAP_READ_BIT`, `GL_MAP_WRITE_BIT`, `GL_MAP_PERSISTENT_BIT`, `GL_MAP_COHERENT_BIT` and
`GL_CLIENT_STORAGE_BIT`. The `GL_MAP_*_BIT`s are used when we want to call `glMapNamedBuffer` / `glMapNamedBufferRange`, which will give us pointers to the buffers. This can be useful when doing *Persistently Mapped Buffers*, but I won't talk about that concept here. So, tl;dr, `GL_DYNAMIC_STORAGE_BIT` will be fine for the most cases.
To make the buffer size truly constant, we will only call `glNamedBufferStorage` followed by *only* using `glMapNamedBuffer` / `glMapNamedBufferRange` or `glNamedBufferSubData`.
While in most cases, this is the optimal solution, some drivers on some vendors have a slow path when using `glBufferSubData` / `glNamedBufferSubData`, so in that case, a staging buffer followed by
[`glCopyNamedBufferSubData`](https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glCopyBufferSubData.xhtml) might be a more optimal choice. Code taken from [my model loading example](https://github.com/rtryan98/OpenGL/blob/master/examples/Models/src/Main.cpp):
{% highlight cpp %}
int32_t main()
{
    // ...
    // create the vertex buffer and the index buffer
    // store the vertex and index data into constant,
    // from the client (cpu) inaccessible buffers.
    uint32_t ibo{}, vbo{};
    glCreateBuffers(1, &ibo);
    glNamedBufferStorage(ibo, iboData.size() * sizeof(uint32_t), iboData.data(), 0x0);
    glCreateBuffers(1, &vbo);
    glNamedBufferStorage(vbo, vboData.size() * sizeof(Vertex), vboData.data(), 0x0);
    // ...
}
{% endhighlight %}

## Vertex Array Objects
As with the buffers, we will replace `glGenVertexArrays` and `glBindVertexArray` with `glCreateVertexArrays`. Again, the signature is the same between `glGen*` and `glCreate*`.
Now to the new stuff. [`glVertexArrayAttrib*Format(GLuint vaobj, GLuint attribindex, GLint size, GLenum type, GLboolean normalized, GLuint relativeoffset)`](https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glVertexAttribFormat.xhtml) is the DSA-equivalent of [`glVertexAttribPointer`](https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glVertexAttribPointer.xhtml). The signatures are really close to oneanother, so we can pretty much translate them directly. (The *`normalized`* parameter is not available for the `glVertexArrayAttribIFormat` and `glVertexArrayAttribLFormat` functions.)
Only difference being that we have to specify the ID of our VAO first. That is not all though, now we have to specify the *binding point* for this format. This is used to be able to swap buffers bound to the VAO
without the need of calling `glVertexAttribPointer` again. This is done with something as simple as calling [`glVertexArrayAttribBinding(GLuint vaobj, GLuint attribindex, GLuint bindingindex)`](https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glVertexAttribBinding.xhtml). Here we specify our vao first, followed by the `attribindex` (Otherwise known as the `location` in the Vertex Shader for our `in` attributes) and
the *`bindingindex`*. Assuming we have only one Vertex Buffer, with packed data, then this can always be 0. But assuming we have data in different buffers, like we have one buffer for positions, one for normals, then
we want to set the *`bindingindex`* to something different for each attribute. That will allow us to specifically "map" buffers to locations in our VAO, with only a single call, compared to before, where we had to do
all the formatting again of the VAO. The only thing left is to enable the Attrib. It's as simple as calling [`glEnableVertexArrayAttrib(GLuint vaobj, GLuint index)`](https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glEnableVertexAttribArray.xhtml). This is basically the same as in non-DSA, except that we specify the VAO here. A full initialization of VAO for packed Vertex Data would look like that:
{% highlight cpp %}
struct Vertex
{
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 tc;
}

int32_t main()
{
    // ...
    // create the VAO
    uint32_t vao{};
    glCreateVertexArrays(1, &vao);

    // enable the first attrib, set the format and bind it to `0`
    glEnableVertexArrayAttrib(vao, 0);
    glVertexArrayAttribFormat(vao, 0, 3, GL_FLOAT, GL_FALSE, offsetof(Vertex, position));
    glVertexArrayAttribBinding(vao, 0, 0);
    
    // enable the second attrib, set the format and bind it to `0`
    glEnableVertexArrayAttrib(vao, 1);
    glVertexArrayAttribFormat(vao, 1, 3, GL_FLOAT, GL_FALSE, offsetof(Vertex, normal));
    glVertexArrayAttribBinding(vao, 1, 0);
    
    // enable the third attrib, set the format and bind it to `0`
    glEnableVertexArrayAttrib(vao, 2);
    glVertexArrayAttribFormat(vao, 2, 2, GL_FLOAT, GL_FALSE, offsetof(Vertex, tc));
    glVertexArrayAttribBinding(vao, 2, 0);

    // ...
}
{% endhighlight %}
We're not finished here. We now want to use the buffer. To do that, we only need to call [`glVertexArrayVertexBuffer(GLuint vaobj, GLuint bindingindex, GLuint buffer, GLintptr offset, GLsizei stride)`](https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glBindVertexBuffer.xhtml). As you can see, *`bindingindex`* makes a return here. It is really useful, because we can now simply re-bind different buffers to the same VAO,
ultimately allowing us to only use one VAO per Vertex-Layout! Ofc. for optimization purposes we would still use different *`bindingindex`*es (for example when doing shadows), but for the purpose of learning, we won't here. We gain even more control than with the previously used `glBindBuffer(GLenum target, GLuint buffer)`. We can now bind a *buffer range* to the VAO. That means we can very easily have fewer, bigger allocations containing more data than before. We could even go as far as to creating a GPU memory allocator for OpenGL. Though I won't do that here. To wrap it up for VAOs, we're still missing our trusty
index buffer. This is even simpler. A single call to [glVertexArrayElementBuffer(GLuint vao, GLuint buffer)](https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glVertexArrayElementBuffer.xhtml) will suffice.
The signature should be self explanatory. That will extend our example with the following:
{% highlight cpp %}
    // ...
    glVertexArrayVertexBuffer(vao, 0, vbo, 0, sizeof(Vertex));
    glVertexArrayElementBuffer(vao, ibo);
    // ...
{% endhighlight %}

## Textures
The DSA equivalent to `glGenTextures` and `glBindTexture` is `glCreateTextures`. This time, the signature is not the same. We do have to specify the texture target. That means,
{% highlight cpp %}
glGenTextures(1, &texture);
glBindTextre(GL_TEXTURE_2D, texture);
{% endhighlight %}
becomes
{% highlight cpp %}
glCreateTextures(GL_TEXTURE_2D, 1, &texture);
{% endhighlight %}
Other than that, `glTextureParameteri` is the equivalent to `glTexParameteri`. The usage is the same, with the difference of specifying your texture ID instead of the texture target.
Also, instead of using `glTexImage`, we will now use `glTextureStorage*D` and `glTextureSubImage*D`. In a 2D-Texture example, that'd be [`glTextureStorage2D`](https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glTexStorage2D.xhtml).
The list of viable *`internalformat`*s is long, so I recommend you to use the official specification. In most cases, `GL_RGBA8` will suffice though.
To reduce our calls even further, we will no longer be using `glActiveTexture`, but [`glBindTextureUnit(GLuint unit, GLuint texture)`](https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glBindTextureUnit.xhtml).
This will change the following:
{% highlight cpp %}
// assume `n` is an uint32_t between 0 and 32.
// before
glActiveTexture(GL_TEXTURE_0 + n);
glBindTexture(GL_TEXTURE_2D, texture);
// after
glBindTextureUnit(n, texture);
{% endhighlight %}
Lastly, a few things. To generate mipmaps, we will call [`glGenerateTextureMipmap`](https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glGenerateMipmap.xhtml) instead of `glGenerateMipmap`.
To upload cubemaps, we will use [`glTextureSubImage3D`](https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glTexSubImage3D.xhtml). Our *`zoffset`* parameter is the current face in the cubemap.
And finally, to `glTexImage`. This function has no DSA equivalent. It is better to use immutable textures and that's why it was decided not to implement that function. Consider only using `glTextureStorage*D`
and `glTextureSubImage*D`.

## Framebuffers
`glCreateFramebuffers` has the same signature as `glGenFramebuffers`. Nothing much has changed with framebuffers except that they are easier to use with DSA.
[`glBlitNamedFramebuffer(GLuint readFramebuffer, GLuint drawFramebuffer, ...)`](https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glBlitFramebuffer.xhtml)
is pretty much the same as `glBlitFramebuffer`, except that we don't have to specifically bind them to `GL_READ_FRAMEBUFFER` and `GL_WRITE_FRAMEBUFFER`.
`glNamedFramebufferTexture` replaces `glFramebufferTexture`, `glFramebufferTexture1D`, `glFramebufferTexture2D` and `glFramebufferTexture3D`. The usage is pretty much the same,
except that you now pass in the FBO ID instead of specifying your target. To check if the framebuffer is valid, you now use [`glCheckNamedFramebufferStatus`](https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glCheckFramebufferStatus.xhtml).
[`glClearNamedFramebuffer`](https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glClearBuffer.xhtml) is a bit different on the other hand, if you're used to `glClear`.
A quick replacement for clearing a framebuffer (one color texture, one depth texture) would be:
{% highlight cpp %}
    // if you haven't created a FBO, the fbo param can be 0 to achieve the same effect.
    // before
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // after
    float_t clearColor[] =
    {
        0.0f, 0.0f, 0.0f, 1.0f
    }
    float_t depth{ 1.0f };
    glClearNamedFramebufferfv(fbo, GL_COLOR, 0, clearColor);
    glClearNamedFramebufferfv(fbo, GL_DEPTH, 0, &depth);
{% endhighlight %}

## Sources
* [Official Khronos OpenGL Wiki article on DSA](https://www.khronos.org/opengl/wiki/Direct_State_Access)
* [A Guide to Modern OpenGL Functions](https://github.com/fendevel/Guide-to-Modern-OpenGL-Functions)
