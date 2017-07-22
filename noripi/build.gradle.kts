import org.gradle.api.tasks.wrapper.Wrapper
import org.gradle.plugins.ide.idea.IdeaPlugin
import org.jetbrains.kotlin.gradle.plugin.KotlinPluginWrapper

buildscript {
    var kotlinVersion: String? by extra; kotlinVersion = "1.1.3"

    repositories {
        mavenCentral()
        maven { setUrl("https://repo.maven.apache.org/maven2") }
        maven { setUrl("https://repo.gradle.org/gradle/repo") }
    }

    dependencies {
        classpath("org.jetbrains.kotlin:kotlin-gradle-plugin:$kotlinVersion")
    }
}

apply {
    plugin<KotlinPluginWrapper>()
    plugin<IdeaPlugin>()
}

repositories {
    mavenCentral()
    jcenter()
}

dependencies {
    val kotlinVersion: String? by extra

    compileOnly(gradleApi())
    compile("org.jetbrains.kotlin:kotlin-stdlib:$kotlinVersion")
    compile("org.jetbrains.kotlinx:kotlinx-support-jdk8:0.3")
}

/***********************************************************************************************
 * TASK DEFINITIONS
 ***********************************************************************************************/
val wrapper = task("wrapper", Wrapper::class) {
    gradleVersion = "3.3"
}
