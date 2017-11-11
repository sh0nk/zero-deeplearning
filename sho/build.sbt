name := "sho"

version := "1.0"

scalaVersion := "2.12.2"


libraryDependencies  ++= Seq(
  // Last stable release
  "org.scalanlp" %% "breeze" % "0.13.1",

  // Native libraries are not included by default. add this if you want them (as of 0.7)
  // Native libraries greatly improve performance, but increase jar sizes.
  // It also packages various blas implementations, which have licenses that may or may not
  // be compatible with the Apache License. No GPL code, as best I know.
  "org.scalanlp" %% "breeze-natives" % "0.13.1",

  // The visualization library is distributed separately as well.
  // It depends on LGPL code
  "org.scalanlp" %% "breeze-viz" % "0.13.1",

  // https://github.com/sh0nk/matplotlib4j
  "com.github.sh0nk" % "matplotlib4j" % "0.3.0"

  //  "com.quantifind" %% "wisp" % "0.0.4"
)


resolvers += Resolver.mavenLocal
resolvers += "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"

updateOptions := updateOptions.value.withLatestSnapshots(false)
